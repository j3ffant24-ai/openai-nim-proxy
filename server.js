// server.js - OpenAI to Groq API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Groq API configuration
const GROQ_API_BASE = 'https://api.groq.com/openai/v1';
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false; // Set to true to show <think> tags in output

// Model mapping — Groq model strings
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'llama-3.1-8b-instant',
  'gpt-4':         'llama-3.3-70b-versatile',
  'gpt-4-turbo':   'deepseek-r1-distill-llama-70b',
  'gpt-4o':        'openai/gpt-oss-120b',
  'claude-3-opus': 'llama-3.3-70b-versatile',
  'claude-3-sonnet':'qwen-qwq-32b',
  'gemini-pro':    'llama-4-scout-17b-16e-instruct',
  'minimax':       'llama-3.3-70b-versatile'
};

// Trim old messages to avoid payload too large errors
const trimMessages = (messages, maxTokens = 24000) => {
  const estimate = msgs => msgs.reduce((sum, m) => sum + Math.ceil((m.content || '').length / 4), 0);
  if (estimate(messages) <= maxTokens) return messages;
  const system = messages.filter(m => m.role === 'system');
  const rest   = messages.filter(m => m.role !== 'system');
  while (rest.length > 1 && estimate([...system, ...rest]) > maxTokens) rest.shift();
  console.warn(`Trimmed messages to ${estimate([...system, ...rest])} estimated tokens`);
  return [...system, ...rest];
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to Groq Proxy',
    reasoning_display: SHOW_REASONING
  });
});

// Test all mapped models — visit /test-models to see which ones work
app.get('/test-models', async (req, res) => {
  const results = {};
  for (const [alias, groqModel] of Object.entries(MODEL_MAPPING)) {
    try {
      const r = await axios.post(`${GROQ_API_BASE}/chat/completions`, {
        model: groqModel,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        stream: false
      }, {
        headers: { 'Authorization': `Bearer ${GROQ_API_KEY}`, 'Content-Type': 'application/json' },
        timeout: 15000,
        validateStatus: () => true
      });
      results[alias] = { groq_model: groqModel, status: r.status, ok: r.status < 400 };
    } catch (err) {
      results[alias] = { groq_model: groqModel, status: 'timeout', ok: false };
    }
  }
  res.json(results);
});

// Models list endpoint (required by WyvernChat and most frontends)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'groq-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream,
            frequency_penalty, presence_penalty, top_p, repetition_penalty } = req.body;

    // Model selection with fallback
    let groqModel = MODEL_MAPPING[model];
    if (!groqModel) {
      const m = model.toLowerCase();
      if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('large')) {
        groqModel = 'llama-3.3-70b-versatile';
      } else if (m.includes('claude') || m.includes('gemini') || m.includes('medium')) {
        groqModel = 'llama-3.3-70b-versatile';
      } else {
        groqModel = 'llama-3.1-8b-instant';
      }
    }

    // Force streaming to prevent 504 timeouts
    const useStream = true;

    // Build request — only include optional params if sent
    const groqRequest = {
      model: groqModel,
      messages: trimMessages(messages),
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 2048,
      stream: useStream
    };

    if (frequency_penalty != null) groqRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty  != null) groqRequest.presence_penalty  = presence_penalty;
    if (top_p             != null) groqRequest.top_p             = top_p;

    // Retry helper with exponential backoff for 429s and 504s
    const groqFetch = async (retries = 6, delay = 3000) => {
      for (let i = 0; i <= retries; i++) {
        try {
          return await axios.post(`${GROQ_API_BASE}/chat/completions`, groqRequest, {
            headers: {
              'Authorization': `Bearer ${GROQ_API_KEY}`,
              'Content-Type': 'application/json'
            },
            responseType: 'stream',
            timeout: 0
          });
        } catch (err) {
          const status = err.response?.status;
          if ((status === 429 || status === 504) && i < retries) {
            const retryAfter = parseInt(err.response?.headers?.['retry-after'] || 0) * 1000;
            const wait = retryAfter || delay * Math.pow(2, i);
            console.warn(`${status} error. Retrying in ${wait}ms... (attempt ${i + 1}/${retries})`);
            await new Promise(r => setTimeout(r, wait));
          } else {
            throw err;
          }
        }
      }
    };

    const response = await groqFetch();

    if (stream) {
      // Pass stream directly to client
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '', reasoningStarted = false, tokenCount = 0, streamDone = false;
      const MAX_STREAM_TOKENS = 2500;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { res.write(line + '\n'); return; }
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
            if (streamDone) return;
            res.write(`data: ${JSON.stringify(data)}\n\n`);
            tokenCount += (data.choices?.[0]?.delta?.content || '').length / 4;
            if (tokenCount > MAX_STREAM_TOKENS) {
              streamDone = true;
              if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
              response.data.destroy();
            }
          } catch (e) { res.write(line + '\n'); }
        });
      });

      response.data.on('end', () => { if (!res.writableEnded) res.end(); });
      response.data.on('error', (err) => { console.error('Stream error:', err); if (!res.writableEnded) res.end(); });

    } else {
      // Collect stream and return as JSON for non-streaming clients
      let buffer = '', fullContent = '', fullReasoning = '', finishReason = '', promptTokens = 0, completionTokens = 0;

      await new Promise((resolve, reject) => {
        response.data.on('data', (chunk) => { buffer += chunk.toString(); });
        response.data.on('end', () => {
          buffer.split('\n').forEach(line => {
            if (!line.startsWith('data: ') || line.includes('[DONE]')) return;
            try {
              const data = JSON.parse(line.slice(6));
              fullContent   += data.choices?.[0]?.delta?.content           || '';
              fullReasoning += data.choices?.[0]?.delta?.reasoning_content || '';
              if (data.choices?.[0]?.finish_reason) finishReason = data.choices[0].finish_reason;
              if (data.usage) { promptTokens = data.usage.prompt_tokens || 0; completionTokens = data.usage.completion_tokens || 0; }
            } catch (e) {}
          });
          resolve();
        });
        response.data.on('error', reject);
      });

      if (SHOW_REASONING && fullReasoning) fullContent = '<think>\n' + fullReasoning + '\n</think>\n\n' + fullContent;

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: fullContent }, finish_reason: finishReason }],
        usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens }
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    if (res.headersSent) {
      if (!res.writableEnded) res.end();
    } else {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to Groq Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});