/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';

export type OpenRouterContentGeneratorConfig = {
  apiKey: string;
  baseUrl?: string;
  tools?: any[];
};

export class OpenRouterContentGenerator implements ContentGenerator {
  private apiKey: string;
  private baseUrl: string;
  private openai: OpenAI;

  constructor(config: OpenRouterContentGeneratorConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://openrouter.ai/api/v1';
    this.openai = new OpenAI({
      apiKey: this.apiKey,
      baseURL: this.baseUrl,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    // userPromptId: string,
  ): Promise<GenerateContentResponse> {
    // Convert Gemini format to OpenAI format
    const messages = this.convertGeminiToOpenAI(request.contents as any);
    const model = (request as any).model || 'openrouter/auto';

    const params: any = {
      model,
      messages,
    };

    if (request.config) {
      const config = request.config;
      if (config.maxOutputTokens) params.max_tokens = config.maxOutputTokens;
      if (config.temperature) params.temperature = config.temperature;
      if (config.topP) params.top_p = config.topP;
      if (config.tools) {
        params.tools = this.convertGeminiToolsToOpenAI(config.tools);
      }
    }

    const completion = await this.openai.chat.completions.create(params);

    // Convert OpenAI format back to Gemini format
    return this.convertOpenAIToGemini(completion);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    // userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    // For simplicity, implement non-streaming first
    const result = await this.generateContent(request);
    async function* generator() {
      yield result;
    }
    return generator();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // OpenRouter doesn't have a direct token count API, approximate
    const text = JSON.stringify((request as any).contents);
    const tokens = Math.ceil(text.length / 4); // Rough approximation
    return { totalTokens: tokens };
  }

  async embedContent() // request: EmbedContentParameters,
  : Promise<EmbedContentResponse> {
    // OpenRouter doesn't support embeddings directly, return empty
    return { embeddings: [] };
  }

  private convertGeminiToOpenAI(contents: any[]): any[] {
    // Simple conversion: assume contents are text
    return contents.map((content) => ({
      role: content.role === 'user' ? 'user' : 'assistant',
      content: content.parts?.map((part: any) => part.text).join('') || '',
    }));
  }

  private convertGeminiToolsToOpenAI(tools: any[]): any[] {
    const openaiTools: any[] = [];
    for (const tool of tools) {
      if (tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          openaiTools.push({
            type: 'function',
            function: {
              name: func.name,
              description: func.description,
              parameters: func.parametersJsonSchema || func.parameters,
            },
          });
        }
      } else if (tool.functionDeclaration) {
        openaiTools.push({
          type: 'function',
          function: {
            name: tool.functionDeclaration.name,
            description: tool.functionDeclaration.description,
            parameters: tool.functionDeclaration.parameters,
          },
        });
      }
    }
    return openaiTools;
  }

  private convertOpenAIToGemini(
    completion: OpenAI.Chat.Completions.ChatCompletion,
  ): GenerateContentResponse {
    const message = completion.choices[0].message;
    let parts: any[] = [];

    if (message.tool_calls) {
      parts = message.tool_calls.map((tc) => ({
        functionCall: {
          name: (tc as any).function.name,
          args: JSON.parse((tc as any).function.arguments),
        },
      }));
    } else if (message.content) {
      parts = [{ text: message.content }];
    }

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: 'STOP' as any,
          index: 0,
        },
      ],
      usageMetadata: {
        promptTokenCount: completion.usage?.prompt_tokens || 0,
        candidatesTokenCount: completion.usage?.completion_tokens || 0,
        totalTokenCount: completion.usage?.total_tokens || 0,
      },
    } as any;
  }
}
