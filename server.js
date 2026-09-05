// server.js - OpenAI to Google Gemini API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Gemini API configuration
const GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1beta/openai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Model mapping — Gemini model strings
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'gemini-3.1-flash-lite',
  'gpt-4':         'gemini-3.8-flash',
  'gpt-4-turbo':   'gemini-3.1-flash-lite',
  'gpt-4o':        'gemini-3.1-flash-lite',
  'claude-3-opus': 'gemini-3.8-flash',
  'claude-3-sonnet':'gemini-3.1-flash-lite',
  'gemini-pro':    'gemini-3.1-flash-lite',
  'minimax':       'gemini-3.1-flash-lite'
};

// Rate limiter — keeps requests under Gemini's RPM caps
const lastRequest = { flash: 0, lite: 0 };
const waitForSlot = async (model) => {
  const isLite   = model.includes('lite');
  const spacing  = isLite ? 2100 : 4300; // 28 RPM for Lite, 14 RPM for Flash
  const lastTime = isLite ? lastRequest.lite : lastRequest.flash;
  const wait     = Math.max(0, spacing - (Date.now() - lastTime));
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
  if (isLite) lastRequest.lite = Date.now();
  else        lastRequest.flash = Date.now();
};

// Trim old messages — Gemini has 1M context so limit is generous
const trimMessages = (messages, maxTokens = 16000) => {
  const estimate = msgs => msgs.reduce((sum, m) => sum + Math.ceil((m.content || '').length / 4), 0);
  if (estimate(messages) <= maxTokens) return messages;
  const system = messages.filter(m => m.role === 'system');
  const rest   = messages.filter(m => m.role !== 'system');
  while (rest.length > 1 && estimate([...system, ...rest]) > maxTokens) rest.shift();
  console.warn(`Trimmed messages to ${estimate([...system, ...rest])} estimated tokens`);
  return [...system, ...rest];
};

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to Gemini Proxy' });
});

// Test all mapped models — visit /test-models to see which ones work
app.get('/test-models', async (req, res) => {
  const results = {};
  for (const [alias, geminiModel] of Object.entries(MODEL_MAPPING)) {
    try {
      const r = await axios.post(`${GEMINI_API_BASE}/chat/completions`, {
        model: geminiModel,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        stream: false
      }, {
        headers: { 'Authorization': `Bearer ${GEMINI_API_KEY}`, 'Content-Type': 'application/json' },
        timeout: 15000,
        validateStatus: () => true
      });
      results[alias] = { gemini_model: geminiModel, status: r.status, ok: r.status < 400 };
    } catch (err) {
      results[alias] = { gemini_model: geminiModel, status: 'timeout', ok: false };
    }
  }
  res.json(results);
});

// Models list — required by WyvernChat and most frontends
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'gemini-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Individual model lookup — some frontends call GET /v1/models/:id to verify
app.get('/v1/models/:modelId', (req, res) => {
  res.json({
    id: req.params.modelId,
    object: 'model',
    created: 1700000000,
    owned_by: 'gemini-proxy'
  });
});

// Chat completions — main proxy
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream,
            frequency_penalty, presence_penalty, top_p } = req.body;

    // Model selection with fallback
    let geminiModel = MODEL_MAPPING[model];
    if (!geminiModel) {
      const m = model.toLowerCase();
      geminiModel = (m.includes('gpt-4') || m.includes('large') || m.includes('opus'))
        ? 'gemini-2.5-flash'
        : 'gemini-2.5-flash-lite';
    }

    // Force streaming to keep Render connection alive
    const useStream = true;

    // Build request — strip unsupported Gemini params
    const geminiRequest = {
      model: geminiModel,
      messages: trimMessages(messages),
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 2048,
      stream: useStream
    };

    // Only forward top_p — Gemini rejects frequency/presence_penalty
    if (top_p != null) geminiRequest.top_p = top_p;

    // Retry with exponential backoff for 429 and 503
    const geminiFetch = async (retries = 6, delay = 3000) => {
      for (let i = 0; i <= retries; i++) {
        try {
          return await axios.post(`${GEMINI_API_BASE}/chat/completions`, geminiRequest, {
            headers: {
              'Authorization': `Bearer ${GEMINI_API_KEY}`,
              'Content-Type': 'application/json'
            },
            responseType: 'stream',
            timeout: 0
          });
        } catch (err) {
          const status = err.response?.status;
          if ((status === 429 || status === 503) && i < retries) {
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

    // Throttle to stay under Gemini's RPM limits
    await waitForSlot(geminiModel);

    const response = await geminiFetch();

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '', tokenCount = 0, streamDone = false;
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
      // Collect stream and return as JSON
      let buffer = '', fullContent = '', finishReason = '';

      await new Promise((resolve, reject) => {
        response.data.on('data', (chunk) => { buffer += chunk.toString(); });
        response.data.on('end', () => {
          buffer.split('\n').forEach(line => {
            if (!line.startsWith('data: ') || line.includes('[DONE]')) return;
            try {
              const data = JSON.parse(line.slice(6));
              fullContent += data.choices?.[0]?.delta?.content || '';
              if (data.choices?.[0]?.finish_reason) finishReason = data.choices[0].finish_reason;
            } catch (e) {}
          });
          resolve();
        });
        response.data.on('error', reject);
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: fullContent }, finish_reason: finishReason }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
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
  console.log(`OpenAI to Gemini Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});