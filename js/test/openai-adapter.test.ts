import { describe, it, expect } from "vitest";
import { OpenAIAdapter } from "../src/capture/openai.js";

const a = new OpenAIAdapter();

describe("OpenAIAdapter chat completions", () => {
  it("extracts tool schemas first (as tool_schema/system JSON), then messages", () => {
    const blocks = a.extractBlocks({
      model: "gpt-4o",
      tools: [{ type: "function", function: { name: "get_weather" } }],
      messages: [
        { role: "system", content: "be terse" },
        { role: "user", content: "hi" },
      ],
    });
    expect(blocks.map((b) => [b.role, b.kind])).toEqual([
      ["system", "tool_schema"],
      ["system", "message"],
      ["user", "message"],
    ]);
    // tool schema serialized with sorted keys (stable JSON)
    expect(blocks[0].text).toBe(
      '{"function": {"name": "get_weather"}, "type": "function"}',
    );
    expect(blocks[1].text).toBe("be terse");
    expect(blocks[2].text).toBe("hi");
  });

  it("splits list content into one content_part per part", () => {
    const blocks = a.extractBlocks({
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "input_audio", input_audio: { data: "AAA=", format: "wav" } },
          ],
        },
      ],
    });
    // Both parts keep the stable-JSON `content_part` path. An `image_url` part
    // is the one that leaves it — it becomes an 'image' block instead; see
    // test/images.test.ts.
    expect(blocks.map((b) => b.kind)).toEqual(["content_part", "content_part"]);
    expect(blocks[0].text).toBe('{"text": "look", "type": "text"}');
  });

  it("emits assistant tool_calls as content_part and skips the empty message", () => {
    const blocks = a.extractBlocks({
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "c1", type: "function", function: { name: "f" } }],
        },
      ],
    });
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe("content_part");
    expect(blocks[0].role).toBe("assistant");
  });
});

describe("OpenAIAdapter responses API", () => {
  it("extracts instructions -> system, tools -> tool_schema, input string -> user", () => {
    const blocks = a.extractBlocks({
      model: "gpt-4o",
      instructions: "you are a bot",
      tools: [{ type: "function", name: "search" }],
      input: "what is the weather",
    });
    expect(blocks.map((b) => [b.role, b.kind])).toEqual([
      ["system", "message"],
      ["system", "tool_schema"],
      ["user", "message"],
    ]);
    expect(blocks[0].text).toBe("you are a bot");
    expect(blocks[2].text).toBe("what is the weather");
  });

  it("splits an input list into per-item / per-part blocks", () => {
    const blocks = a.extractBlocks({
      input: [
        {
          role: "user",
          content: [{ type: "input_text", text: "hello" }],
        },
        { type: "function_call", name: "f", arguments: "{}" },
        { type: "function_call_output", output: "42" },
      ],
    });
    expect(blocks.map((b) => [b.role, b.kind])).toEqual([
      ["user", "content_part"],
      ["assistant", "content_part"],
      ["tool", "content_part"],
    ]);
    // input_text part uses its own text field
    expect(blocks[0].text).toBe("hello");
  });
});

describe("OpenAIAdapter extractParams", () => {
  it("drops chat content keys but keeps model + sampling", () => {
    const p = a.extractParams({
      model: "gpt-4o",
      temperature: 0.7,
      messages: [{ role: "user", content: "hi" }],
      tools: [{ type: "function" }],
    });
    expect(p).toEqual({ model: "gpt-4o", temperature: 0.7 });
  });

  it("drops responses content keys but KEEPS previous_response_id", () => {
    const p = a.extractParams({
      model: "gpt-4o",
      instructions: "x",
      input: "y",
      tools: [],
      previous_response_id: "resp_123",
    });
    expect(p).toEqual({ model: "gpt-4o", previous_response_id: "resp_123" });
  });
});

describe("OpenAIAdapter extractUsage (both families)", () => {
  it("reads the chat prompt_tokens family", () => {
    expect(
      a.extractUsage({
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }),
    ).toEqual({ prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 });
  });

  it("reads the responses input_tokens family", () => {
    expect(
      a.extractUsage({
        usage: { input_tokens: 8, output_tokens: 3, total_tokens: 11 },
      }),
    ).toEqual({ input_tokens: 8, output_tokens: 3, total_tokens: 11 });
  });

  it("returns null when there is no usage", () => {
    expect(a.extractUsage({})).toBeNull();
    expect(a.extractUsage(null)).toBeNull();
  });
});

describe("OpenAIAdapter accumulateStreamUsage", () => {
  it("folds a chat final chunk's usage", () => {
    const state: Record<string, unknown> = {};
    a.accumulateStreamUsage(
      { object: "chat.completion.chunk", choices: [], usage: null },
      state,
    );
    expect(state).toEqual({});
    a.accumulateStreamUsage(
      {
        object: "chat.completion.chunk",
        choices: [],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      },
      state,
    );
    expect(state).toEqual({
      prompt_tokens: 5,
      completion_tokens: 2,
      total_tokens: 7,
    });
  });

  it("folds a responses response.completed event's usage", () => {
    const state: Record<string, unknown> = {};
    a.accumulateStreamUsage(
      {
        type: "response.completed",
        response: {
          usage: { input_tokens: 8, output_tokens: 3, total_tokens: 11 },
        },
      },
      state,
    );
    expect(state).toEqual({
      input_tokens: 8,
      output_tokens: 3,
      total_tokens: 11,
    });
  });

  it("never throws on a malformed chunk", () => {
    const state: Record<string, unknown> = {};
    expect(() => a.accumulateStreamUsage(null, state)).not.toThrow();
    expect(() => a.accumulateStreamUsage(42, state)).not.toThrow();
    expect(state).toEqual({});
  });
});
