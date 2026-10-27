import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';
import dotenv from 'dotenv';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// Middleware for parsing large base64 documents (PDFs, images)
app.use(express.json({ limit: '60mb' }));
app.use(express.urlencoded({ extended: true, limit: '60mb' }));

// Shared Gemini client utility on the server
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// In-memory cache for fast session re-processing (LRU bounded)
const analysisCache = new Map<string, any>();
const MAX_CACHE_ITEMS = 60;

function getCacheKey(payload: { fileData?: string; text?: string; mimeType?: string; noticeName?: string }): string {
  const hash = crypto.createHash('sha256');
  if (payload.text) {
    hash.update(`text:${payload.text.trim()}`);
  } else if (payload.fileData) {
    hash.update(`file:${payload.noticeName || ''}:${payload.mimeType || ''}:${payload.fileData.slice(0, 1500)}:${payload.fileData.length}`);
  }
  return hash.digest('hex');
}

// JSON Schema for Notice Analysis output
const analysisSchema = {
  type: Type.OBJECT,
  properties: {
    title: {
      type: Type.STRING,
      description: 'The formal title or subject of the notice.',
    },
    issuingOrganization: {
      type: Type.STRING,
      description: 'The university, board, government body, or department issuing this notice.',
    },
    noticeReferenceNo: {
      type: Type.STRING,
      description: 'Official notice number, circular reference code, or Memo ID if present; otherwise empty string.',
    },
    announcementDate: {
      type: Type.STRING,
      description: 'Date the notice was issued (YYYY-MM-DD or original text if year unstated); empty string if not found.',
    },
    category: {
      type: Type.STRING,
      description: 'Category like Examination, Scholarship, Fee Circular, Admission, Recruitment, or General Notice.',
    },
    simpleSummary: {
      type: Type.OBJECT,
      properties: {
        en: {
          type: Type.OBJECT,
          properties: {
            quickSummary: {
              type: Type.STRING,
              description: 'Crisp, plain-language summary in 3-5 concise bullet points or sentences.',
            },
            whatThisMeansForYou: {
              type: Type.STRING,
              description: 'Direct explanation of how this affects the student or reader in everyday terms.',
            },
            importantInstructions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Key step-by-step instructions the user needs to follow.',
            },
            conditionsAndEligibility: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Eligibility criteria, restrictions, minimum attendance, or prerequisite conditions.',
            },
            requiredDocuments: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'All certificates, IDs, photos, forms, or receipts explicitly required.',
            },
            feesAndPenalties: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Fees, late fee charges, penalty dates, and payment portal details.',
            },
          },
          required: [
            'quickSummary',
            'whatThisMeansForYou',
            'importantInstructions',
            'conditionsAndEligibility',
            'requiredDocuments',
            'feesAndPenalties',
          ],
        },
        hi: {
          type: Type.OBJECT,
          properties: {
            quickSummary: {
              type: Type.STRING,
              description: 'सरल और स्पष्ट हिंदी में 3-5 मुख्य बिंदुओं का सारांश।',
            },
            whatThisMeansForYou: {
              type: Type.STRING,
              description: 'इस नोटिस का आपके लिए क्या अर्थ है, सरल हिंदी में।',
            },
            importantInstructions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'महत्वपूर्ण निर्देश जो आपको पालन करने हैं।',
            },
            conditionsAndEligibility: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'पात्रता शर्तें, न्यूनतम उपस्थिति या नियम।',
            },
            requiredDocuments: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'आवश्यक दस्तावेज, प्रमाण पत्र या रसीदें।',
            },
            feesAndPenalties: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'शुल्क, विलंब शुल्क (लेट फीस) और भुगतान निर्देश।',
            },
          },
          required: [
            'quickSummary',
            'whatThisMeansForYou',
            'importantInstructions',
            'conditionsAndEligibility',
            'requiredDocuments',
            'feesAndPenalties',
          ],
        },
      },
      required: ['en', 'hi'],
    },
    facts: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING },
          category: {
            type: Type.STRING,
            description: 'One of: deadline, exam, fee, document, eligibility, instruction, restriction, contact',
          },
          title: { type: Type.STRING },
          description: { type: Type.STRING },
          sourceQuote: {
            type: Type.STRING,
            description: 'Exact verbatim excerpt or sentence from the notice text supporting this fact.',
          },
          pageOrSection: {
            type: Type.STRING,
            description: 'Section heading, paragraph, or page number where this fact appears.',
          },
          importance: {
            type: Type.STRING,
            description: 'high, medium, or low',
          },
        },
        required: ['id', 'category', 'title', 'description', 'sourceQuote', 'pageOrSection', 'importance'],
      },
    },
    actionItems: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING },
          title: { type: Type.STRING, description: 'Actionable task title (e.g., Pay semester examination fee)' },
          description: { type: Type.STRING },
          dueDate: {
            type: Type.STRING,
            description: 'Standard YYYY-MM-DD date if explicitly confirmed in notice; null if missing or uncertain.',
          },
          dueTime: {
            type: Type.STRING,
            description: 'HH:MM in 24h format if explicitly mentioned (e.g. 17:00 for 5 PM); otherwise null.',
          },
          isAllDay: { type: Type.BOOLEAN },
          isDeadlinePassed: {
            type: Type.BOOLEAN,
            description: 'True if the date explicitly precedes the current year 2026 or is marked passed.',
          },
          isDateAmbiguous: {
            type: Type.BOOLEAN,
            description: 'True if year is missing, conflicting dates are present, or date is uncertain.',
          },
          ambiguityReason: {
            type: Type.STRING,
            description: 'Explanation if the date is ambiguous, contradictory, or missing year.',
          },
          sourceQuote: {
            type: Type.STRING,
            description: 'Exact sentence from document describing this action and date.',
          },
          category: {
            type: Type.STRING,
            description: 'payment, form_submission, document_prep, attendance, verification, or other',
          },
        },
        required: [
          'id',
          'title',
          'description',
          'isAllDay',
          'isDeadlinePassed',
          'isDateAmbiguous',
          'sourceQuote',
          'category',
        ],
      },
    },
    calendarEvents: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING },
          title: { type: Type.STRING, description: 'Calendar event title' },
          description: { type: Type.STRING, description: 'Event description including source context and steps' },
          startDate: { type: Type.STRING, description: 'YYYY-MM-DD' },
          startTime: { type: Type.STRING, description: 'HH:MM or empty string for all-day' },
          endDate: { type: Type.STRING, description: 'YYYY-MM-DD' },
          endTime: { type: Type.STRING, description: 'HH:MM or empty string for all-day' },
          isAllDay: { type: Type.BOOLEAN },
          eventType: { type: Type.STRING, description: 'deadline, exam, submission, meeting, verification' },
          sourceQuote: { type: Type.STRING },
          isAmbiguous: { type: Type.BOOLEAN },
          ambiguityNote: { type: Type.STRING },
        },
        required: ['id', 'title', 'description', 'startDate', 'endDate', 'isAllDay', 'eventType', 'sourceQuote', 'isAmbiguous'],
      },
    },
    originalExtractedText: {
      type: Type.STRING,
      description: 'Clean transcript or key excerpts (up to 300 words) from the document.',
    },
    processingNotes: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: 'Observations regarding readability, missing years, deadlines, or warnings for the user.',
    },
  },
  required: [
    'title',
    'issuingOrganization',
    'category',
    'simpleSummary',
    'facts',
    'actionItems',
    'calendarEvents',
    'originalExtractedText',
    'processingNotes',
  ],
};

// Resilient fast model caller: tries candidate models in sequence if one returns 503/429
async function callResilientFlash(models: string[], requestConfig: any) {
  let lastErr: any;
  for (const model of models) {
    try {
      console.log(`Analyzing notice with model ${model}...`);
      const response = await ai.models.generateContent({
        ...requestConfig,
        model,
      });
      return response;
    } catch (err: any) {
      lastErr = err;
      const isTransient =
        err.status === 503 ||
        err.status === 429 ||
        (err.message && (err.message.includes('high demand') || err.message.includes('unavailable')));
      if (isTransient) {
        console.warn(`Model ${model} unavailable (status ${err.status}), attempting fallback model...`);
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

// Notice Analysis API Route
app.post('/api/gemini/analyze-notice', async (req, res) => {
  try {
    const { fileData, mimeType, text, noticeName } = req.body;

    if (!fileData && (!text || !text.trim())) {
      return res.status(400).json({ error: 'Please provide either a document file (PDF/image) or pasted text.' });
    }

    // 1. Session Cache Check for instant repeat analysis
    const cacheKey = getCacheKey({ fileData, mimeType, text, noticeName });
    if (analysisCache.has(cacheKey)) {
      const cachedResult = analysisCache.get(cacheKey);
      return res.json({ success: true, data: cachedResult, cached: true });
    }

    const systemInstruction = `You are Notice Mate, an expert AI Notice-to-Action Accessibility Assistant.
Your mission:
1. Thoroughly parse official notices, academic circulars, exam schedules, scholarship bulletins, and government notifications.
2. Extract exact facts, dates, deadlines, eligibility criteria, required documents, and actionable steps in a single fast, structured pass.
3. PRESERVE FACTUAL ACCURACY: Never fabricate a missing year, never invent times, never guess unstated facts.
4. DATE RULES:
   - If a date in the notice lacks a year, set isDateAmbiguous to true, provide an explanation in ambiguityReason/ambiguityNote, and DO NOT guess a fake year.
   - If a deadline has already passed (e.g. before October 2026), mark isDeadlinePassed: true.
   - Distinguish announcement issue dates from actual action deadlines and exam dates.
   - For all dates, attach the exact verbatim sourceQuote from the document.
5. SIMPLIFIED EXPLANATIONS:
   - Provide both simple English ('en') and clear Hindi ('hi') explanations.
   - Keep quickSummary concise in 3-5 clear bullet points or short sentences.
   - Avoid bureaucratic jargon; explain terms simply so students and first-time readers can easily understand their next steps.
6. CALENDAR EVENT CANDIDATES:
   - Extract distinct calendar events for real deadlines, exam dates, fee cutoffs, and document verifications.
   - If no specific start/end time is stated in the notice, designate the event as all-day (isAllDay: true) rather than inventing an artificial hour.
7. Return strictly valid JSON adhering to the provided schema.`;

    const parts: any[] = [];
    let extractedTextSource = text ? text.trim() : '';

    // Fast-path: Check if PDF has embedded digital text to avoid slow vision OCR
    let usePdfVision = false;
    let normalizedMime = (mimeType || '').toLowerCase();
    if (normalizedMime === 'image/jpg' || normalizedMime === 'image/pjpeg') {
      normalizedMime = 'image/jpeg';
    }

    if (fileData && normalizedMime === 'application/pdf') {
      try {
        const buffer = Buffer.from(fileData, 'base64');
        const { PDFParse } = await import('pdf-parse');
        const parser = new PDFParse({ data: buffer });
        const textResult = await parser.getText();
        const pdfText = textResult?.text;
        if (pdfText && pdfText.trim().length > 60) {
          extractedTextSource = pdfText.trim();
        } else {
          usePdfVision = true;
        }
      } catch (err) {
        console.warn('PDF digital text extraction skipped, using multimodal vision:', err);
        usePdfVision = true;
      }
    } else if (fileData) {
      usePdfVision = true;
    }

    let modelCandidates: string[];

    if (extractedTextSource) {
      // Fast text-only pathway: 3x-4x faster than vision OCR
      modelCandidates = ['gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-3.8-flash'];
      parts.push({
        text: `Please analyze the following notice text (${noticeName || 'Notice'}):\n\n"""\n${extractedTextSource}\n"""\n\nExtract all critical facts, 3-5 point simplified explanations in English and Hindi, actionable checklist items with exact source sentences, and calendar event candidates according to the schema.`,
      });
    } else {
      // Multimodal vision pathway for images and scanned PDFs
      modelCandidates = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-3.1-flash-lite'];
      parts.push({
        inlineData: {
          mimeType: normalizedMime || 'image/jpeg',
          data: fileData,
        },
      });
      parts.push({
        text: `Please analyze this uploaded document (${noticeName || 'Notice'}). Extract all critical facts, 3-5 point simplified explanations in English and Hindi, actionable checklist items with exact source sentences, and calendar event candidates according to the schema.`,
      });
    }

    // Call Gemini with resilient flash models
    const response = await callResilientFlash(modelCandidates, {
      contents: { parts },
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: analysisSchema as any,
        temperature: 0.1,
      },
    });

    let jsonText = response.text;
    if (!jsonText) {
      throw new Error('No analysis text received from Gemini model.');
    }

    // Robust JSON cleaning to handle any markdown fence artifacts safely
    jsonText = jsonText.trim();
    if (jsonText.startsWith('```json')) {
      jsonText = jsonText.slice(7);
    } else if (jsonText.startsWith('```')) {
      jsonText = jsonText.slice(3);
    }
    if (jsonText.endsWith('```')) {
      jsonText = jsonText.slice(0, -3);
    }
    jsonText = jsonText.trim();

    const parsedData = JSON.parse(jsonText);

    // If we have the full original source text (from pasted text or digital PDF), preserve it directly
    if (extractedTextSource) {
      parsedData.originalExtractedText = extractedTextSource;
    }

    // Cache successful result in memory for instant reuse
    if (analysisCache.size >= MAX_CACHE_ITEMS) {
      const firstKey = analysisCache.keys().next().value;
      if (firstKey) analysisCache.delete(firstKey);
    }
    analysisCache.set(cacheKey, parsedData);

    return res.json({ success: true, data: parsedData });
  } catch (error: any) {
    console.error('Notice Analysis Error:', error);
    return res.status(500).json({
      error: error.message || 'Failed to process notice with Gemini AI.',
    });
  }
});

// Text-to-Speech API Route for Accessibility
app.post('/api/gemini/tts', async (req, res) => {
  try {
    const { text, voiceName = 'Kore', stylePrompt } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text is required for speech generation.' });
    }

    // Limit text length to ~1200 characters per TTS request for responsiveness
    const trimmedText = text.slice(0, 1500);

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash-lite-tts',
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: trimmedText,
              speechMetadata: {
                style: stylePrompt || 'Warm, clear, and reassuring accessibility assistant',
              },
            },
          ],
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: voiceName || 'Kore' },
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) {
      throw new Error('No audio data received from Gemini TTS.');
    }

    return res.json({
      success: true,
      audioBase64: base64Audio,
      mimeType: 'audio/wav',
    });
  } catch (error: any) {
    console.error('Gemini TTS Error:', error);
    return res.status(500).json({
      error: error.message || 'TTS generation failed.',
    });
  }
});

// Setup development or production serving
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Notice Mate server running on port ${PORT}`);
  });
}

startServer();
