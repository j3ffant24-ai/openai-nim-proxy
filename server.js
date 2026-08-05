// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); // 🔧 FIX: raised from default 100kb — prevents 413 on large character cards / lore

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

process.on('uncaughtException', (err) => {
  console.error('Uncaught exception (server stayed up):', err);
});

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = false;

const THINKING_OVERRIDE = {
  'deepseek-ai/deepseek-v4-pro': true,
  'deepseek-ai/deepseek-v4-flash': true,
  'z-ai/glm-5.2': false,
};

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':         'z-ai/glm-5.2',
  'gpt-4-turbo':   'deepseek-ai/deepseek-v4-flash',
  'gpt-4o':        'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet':'openai/gpt-oss-20b',
  'gemini-pro':    'qwen/qwen3-next-80b-a3b-instruct',
  'minimax':       'minimaxai/minimax-m2.7'
};

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

async function callNimWithRetry(nimRequest, maxRetries = 2) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: nimRequest.stream ? 'stream' : 'json',
        timeout: 0
      });
    } catch (error) {
      const status = error.response?.status;
      const isRetryable = status === 503 || status === 502 || status === 504 || status === 429;
      if (!isRetryable || attempt === maxRetries) throw error;
      const retryAfterHeader = error.response?.headers?.['retry-after'];
      const retryAfterMs = retryAfterHeader ? parseInt(retryAfterHeader, 10) * 1000 : null;
      const waitMs = Math.min(retryAfterMs || 3000 * Math.pow(2, attempt), 15000);
      console.log(`NVIDIA returned ${status}, retrying in ${waitMs}ms (attempt ${attempt + 1}/${maxRetries})`);
      await new Promise(r => setTimeout(r, waitMs));
    }
  }
}

app.post('/v1/chat/completions', async (req, res) => {
  let nimModel;
  try {
    const { model, messages, temperature, max_tokens, stream,
            frequency_penalty, presence_penalty, top_p, repetition_penalty } = req.body;

    nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: s => s < 500
        }).then(r => { if (r.status >= 200 && r.status < 300) nimModel = model; });
      } catch (e) {}

      if (!nimModel) {
        const m = model.toLowerCase();
        if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('405b')) nimModel = 'meta/llama-3.1-405b-instruct';
        else if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) nimModel = 'meta/llama-3.1-70b-instruct';
        else nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    const hasThinkingOverride = nimModel in THINKING_OVERRIDE;
    const wantsThinking = hasThinkingOverride ? THINKING_OVERRIDE[nimModel] : ENABLE_THINKING_MODE;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: Math.min(temperature || 0.6, 1.2),
      max_tokens: wantsThinking ? Math.min(max_tokens || 61440, 163840) : Math.min(max_tokens || 20480, 40960),
      frequency_penalty: frequency_penalty ?? 0.4,
      presence_penalty:  presence_penalty  ?? 0.4,
      top_p:             top_p             ?? 0.9,
      ...(repetition_penalty ? { repetition_penalty } : {}),
      stream: stream || false,
      ...(hasThinkingOverride || ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: wantsThinking } } : {})
    };

    const response = await callNimWithRetry(nimRequest);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      const decoder = new StringDecoder('utf8');

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;
        if (line.includes('[DONE]')) { res.write('data: [DONE]\n\n'); return; }
        try {
          const data = JSON.parse(line.slice(6));
          if (data.choices?.[0]?.delta) {
            const reasoning = data.choices[0].delta.reasoning_content;
            const content   = data.choices[0].delta.content;
            if (SHOW_REASONING) {
              let combined = '';
              if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
              else if (reasoning) { combined = reasoning; }
              if (content && reasoningStarted) { combined += '</think>\n\n' + content; reasoningStarted = false; }
              else if (content) { combined += content; }
              if (combined) { data.choices[0].delta.content = combined; delete data.choices[0].delta.reasoning_content; }
            } else {
              data.choices[0].delta.content = content || '';
              delete data.choices[0].delta.reasoning_content;
            }
          }
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
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
        buffer += decoder.end();
        if (buffer) buffer.split('\n').forEach(processLine);
        res.end();
      });
      response.data.on('error', (err) => { console.error('Stream error:', err); res.end(); });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content)
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          return { index: choice.index, message: { role: choice.message.role, content: fullContent }, finish_reason: choice.finish_reason };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    let errorDetail;
    try { errorDetail = JSON.stringify(error.response?.data); }
    catch (e) { errorDetail = '[response body was a stream]'; }
    console.error('Proxy error | model:', nimModel, '| status:', error.response?.status, '| detail:', errorDetail);
    res.status(error.response?.status || 500).json({
      error: { message: error.message || 'Internal server error', type: 'invalid_request_error', code: error.response?.status || 500 }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Reasoning: ${SHOW_REASONING ? 'ON' : 'OFF'} | Thinking: ${ENABLE_THINKING_MODE ? 'ON' : 'OFF'}`);
});