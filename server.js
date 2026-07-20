// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder'); // correctly handles multi-byte UTF-8 chars split across chunks

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔧 FIX: safety net — an uncaught exception anywhere used to crash the whole
// process (this is what was really causing the mystery 502s: an error in the
// logging line below was taking the entire server down, not NVIDIA). This
// keeps the server serving other requests even if something unexpected slips through.
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception (server stayed up):', err);
});

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter for ALL models

// 🔧 FIX: DeepSeek V4 Pro/Flash are "reasoning" models — NVIDIA's API requires
// chat_template_kwargs.thinking to be set for them specifically, or it can return
// malformed/garbled output instead of cleanly separating reasoning from the final
// answer. This is very likely what caused garbled text at the end of responses.
// Add any other reasoning-capable model here if you see the same symptom.
const REASONING_MODELS = new Set([
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
  'z-ai/glm-5.2', // 🔧 FIX: GLM-5.2 thinks by default on NIM — without chat_template_kwargs,
                  // NVIDIA doesn't gate/separate that reasoning trace into reasoning_content,
                  // so it can bleed straight into the visible answer (the garbled text you saw).
]);

// Model mapping (adjust based on available NIM models — verify against YOUR
// account's /v1/models list first, see Step 1.4. Last verified for this account: July 2026.)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'z-ai/glm-5.2',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v4-flash', // was kimi-k2.6 — access-gated for this account, see Troubleshooting
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-instruct' 
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Calls NVIDIA with automatic retry on temporary server errors (503/502/504) —
// free-tier model endpoints occasionally return these under load, and they
// usually succeed on a quick retry rather than needing the user to resend.
async function callNimWithRetry(nimRequest, maxRetries = 2) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: nimRequest.stream ? 'stream' : 'json'
      });
    } catch (error) {
      const status = error.response?.status;
      const isRetryable = status === 503 || status === 502 || status === 504;
      if (!isRetryable || attempt === maxRetries) throw error;
      const waitMs = 1000 * Math.pow(2, attempt); // 1s, then 2s, then 4s
      console.log(`NVIDIA returned ${status}, retrying in ${waitMs}ms (attempt ${attempt + 1}/${maxRetries})`);
      await new Promise(r => setTimeout(r, waitMs));
    }
  }
}

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  let nimModel; // declared here so it's visible in the catch block below
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const isReasoningModel = ENABLE_THINKING_MODE || REASONING_MODELS.has(nimModel);
    const nimRequest = {
      model: nimModel,
      messages: messages,
      // 🔧 FIX: clamp — an excessive temperature can push generation into degenerate output.
      temperature: Math.min(temperature || 0.6, 1.2),
      // 🔧 FIX: reasoning models spend part of this budget on an internal thinking phase
      // BEFORE producing any visible content — a cap too low can exhaust the whole budget
      // on reasoning alone, leaving zero tokens for the actual answer ("No valid content
      // was generated" = reasoning happened, the answer never did). They get more headroom.
      max_tokens: isReasoningModel ? Math.min(max_tokens || 4096, 8192) : Math.min(max_tokens || 1024, 2048),
      stream: stream || false,
      // 🔧 FIX: chat_template_kwargs must be a TOP-LEVEL field in the JSON body NVIDIA
      // receives. "extra_body" is a Python SDK convenience keyword that the SDK flattens
      // before sending — it isn't a real API field. Sending it literally (as earlier
      // versions of this code did) gets rejected with a 400 once this is actually turned on.
      ...(isReasoningModel ? { chat_template_kwargs: { thinking: true } } : {})
    };
    
    // Make request to NVIDIA NIM API (auto-retries on transient 503/502/504)
    const response = await callNimWithRetry(nimRequest);
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      const decoder = new StringDecoder('utf8'); // 🔧 FIX: buffers partial multi-byte chars across chunk boundaries instead of corrupting them

      // 🔧 FIX: extracted so the leftover buffer can be flushed on 'end' too, instead of being silently dropped
      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          res.write('data: [DONE]\n\n'); // 🔧 FIX: was missing the trailing blank line every SSE event needs
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          if (data.choices?.[0]?.delta) {
            const reasoning = data.choices[0].delta.reasoning_content;
            const content = data.choices[0].delta.content;

            if (SHOW_REASONING) {
              let combinedContent = '';

              if (reasoning && !reasoningStarted) {
                combinedContent = '<think>\n' + reasoning;
                reasoningStarted = true;
              } else if (reasoning) {
                combinedContent = reasoning;
              }

              if (content && reasoningStarted) {
                combinedContent += '</think>\n\n' + content;
                reasoningStarted = false;
              } else if (content) {
                combinedContent += content;
              }

              if (combinedContent) {
                data.choices[0].delta.content = combinedContent;
                delete data.choices[0].delta.reasoning_content;
              }
            } else {
              if (content) {
                data.choices[0].delta.content = content;
              } else {
                data.choices[0].delta.content = '';
              }
              delete data.choices[0].delta.reasoning_content;
            }
          }
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
          // 🔧 FIX: previously forwarded the raw, malformed line to the client with the
          // wrong line ending — that's what caused garbled text. Log and drop it instead.
          console.error('Skipped unparseable stream line:', line);
        }
      };

      response.data.on('data', (chunk) => {
        buffer += decoder.write(chunk);
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(processLine);
      });

      response.data.on('end', () => {
        buffer += decoder.end(); // flush any bytes StringDecoder was holding onto
        if (buffer) buffer.split('\n').forEach(processLine); // 🔧 FIX: previously dropped whatever was left here
        res.end();
      });
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    // 🔧 FIX: error.response.data is a raw Node stream (not JSON) when the failed
    // request was a streaming one — JSON.stringify on that throws a circular-structure
    // error, which was crashing the entire server on every streaming error. Guarded now.
    let errorDetail;
    try {
      errorDetail = JSON.stringify(error.response?.data);
    } catch (stringifyError) {
      errorDetail = '[response body was a stream, not JSON — could not log it]';
    }
    console.error('Proxy error | model:', nimModel, '| status:', error.response?.status, '| detail:', errorDetail);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});