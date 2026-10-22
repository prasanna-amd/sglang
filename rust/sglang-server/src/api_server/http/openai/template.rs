//! Chat prompt rendering through `sglang-processor`.
//!
//! The processor owns template loading and the SGLang-parity formatters; this
//! module adapts the server's wire request to it.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use serde_json::Value;
use sglang_processor::dynamo_renderer::deepseek::common::tokens;
use sglang_processor::dynamo_renderer::{OAIChatLikeRequest, TextInput};
use sglang_processor::dynamo_tokenizers::Tokenizer;

pub use sglang_processor::ChatFormatter;
use sglang_processor::TemplateError;

pub(super) use crate::message::request::RenderedChat;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// Encodes rendered chat prompts as Python's chat layer does.
#[derive(Clone)]
pub struct PromptEncoder {
    tokenizer: Tokenizer,
    /// Ids the tokenizer adds on its own (`encode("")`).
    auto_specials: Vec<u32>,
    bos: Option<u32>,
}

impl PromptEncoder {
    pub fn new(tokenizer: Tokenizer, server_args: &crate::message::config::ServerArgs) -> Self {
        use crate::tokenizer_manager::tokenizer::TextTokenizer;
        let bos = crate::tokenizer_manager::tokenizer::DynamoTokenizer::new(
            tokenizer.clone(),
            &server_args.tokenizer_path,
            server_args.revision.as_deref(),
        )
        .bos_token_id()
        .and_then(|id| u32::try_from(id).ok());
        let auto_specials = tokenizer
            .encode("")
            .map(|encoding| encoding.token_ids().to_vec())
            .unwrap_or_default();
        Self {
            tokenizer,
            auto_specials,
            bos,
        }
    }

    /// The template output without the tokenizer's own specials, its segments
    /// keeping their special-token policy (Kimi K3), then the continuation
    /// encoded on its own without its BOS.
    pub(super) fn encode(&self, rendered: &RenderedChat) -> Result<Vec<u32>, String> {
        let encode = |text: &str| match self.tokenizer.encode(text) {
            Ok(encoding) => Ok(encoding.token_ids().to_vec()),
            Err(error) => Err(error.to_string()),
        };
        let mut ids = match rendered.prompt.encode_segments() {
            Some(segments) => self
                .tokenizer
                .encode_segments(&segments)
                .map_err(|error| error.to_string())?
                .token_ids()
                .to_vec(),
            None => encode(rendered.prompt.as_str())?,
        };
        if ids.starts_with(&self.auto_specials) {
            ids.drain(..self.auto_specials.len());
        }
        if !rendered.continuation.is_empty() {
            let suffix = encode(&rendered.continuation)?;
            let skip = usize::from(self.bos.is_some() && suffix.first() == self.bos.as_ref());
            ids.extend(&suffix[skip..]);
        }
        Ok(ids)
    }

    pub(super) fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Render a chat request for the tokenizer pool, returning the
/// `continue_final_message` prefix apart from the prompt. Formatters with an
/// SGLang request path render the raw `body`; the rest render `request`.
pub(super) fn render_chat(
    formatter: &ChatFormatter,
    body: Value,
    request: &CreateChatCompletionRequest,
    kwargs: Option<&ChatTemplateKwargs>,
    defaults: &ChatTemplateKwargs,
) -> Result<RenderedChat, TemplateError> {
    let continuing = body["continue_final_message"] == true;
    let kwargs = template_kwargs(&body, kwargs, defaults);
    match render_body(formatter, body, defaults) {
        // There is no engine to hand a request the SGLang path declines to,
        // so the formatter's own template renders it, as it did before that
        // path existed. DeepSeek-V4 has no other path.
        Some(Err(TemplateError::EngineOnly { .. }))
            if !matches!(formatter, ChatFormatter::DeepSeekV4(_)) =>
        {
            render_template(formatter, request, &kwargs, continuing)
        }
        Some(rendered) => rendered,
        None => render_template(formatter, request, &kwargs, continuing),
    }
}

/// Render the raw body as SGLang does, retaining the continuation boundary.
pub(super) fn render_body(
    formatter: &ChatFormatter,
    body: Value,
    defaults: &ChatTemplateKwargs,
) -> Option<Result<RenderedChat, TemplateError>> {
    if !matches!(
        formatter,
        ChatFormatter::DeepSeekV4(_)
            | ChatFormatter::DeepSeekV41
            | ChatFormatter::Glm53(_)
            | ChatFormatter::KimiK3
            | ChatFormatter::Qwen38 { .. }
    ) {
        return None;
    }
    Some(
        formatter
            .render_request(body, defaults)
            .map(|(prompt, mut continuation)| {
                if matches!(
                    formatter,
                    ChatFormatter::DeepSeekV4(_) | ChatFormatter::DeepSeekV41
                ) && let Some(rest) = continuation.strip_prefix(tokens::BOS)
                {
                    continuation = rest.to_owned();
                }
                RenderedChat {
                    prompt,
                    continuation,
                }
            }),
    )
}

/// Python's template kwargs: the server defaults, under the request's effort
/// and the thinking toggle it implies (`ChatCompletionRequest`'s validator),
/// under the request's own `chat_template_kwargs`.
fn template_kwargs(
    body: &Value,
    kwargs: Option<&ChatTemplateKwargs>,
    defaults: &ChatTemplateKwargs,
) -> ChatTemplateKwargs {
    let mut merged = defaults.clone();
    if let Some(effort) = sglang_processor::requested_effort(body) {
        merged.insert("reasoning_effort".into(), effort.clone());
    }
    if let Some(thinking) = sglang_processor::requested_thinking(body) {
        merged.insert("thinking".into(), thinking.into());
        merged.insert("enable_thinking".into(), thinking.into());
    }
    merged.extend(kwargs.cloned().unwrap_or_default());
    merged
}

/// Render `request` through the formatter's template (Jinja, Dynamo's
/// encoder, or the legacy conversation).
fn render_template(
    formatter: &ChatFormatter,
    request: &CreateChatCompletionRequest,
    kwargs: &ChatTemplateKwargs,
    continuing: bool,
) -> Result<RenderedChat, TemplateError> {
    let (messages, continuation) = match formatter {
        ChatFormatter::Legacy(_) => None,
        _ => last_assistant_message(request, continuing),
    }
    .map_or((None, String::new()), |(messages, prefix)| {
        (Some(messages), prefix)
    });
    let request = TemplateRequest {
        request,
        kwargs: (!kwargs.is_empty()).then_some(kwargs),
        messages,
    };
    formatter
        .render_prompt(&request)
        .map(|prompt| RenderedChat {
            prompt,
            continuation,
        })
}

/// Python's `_handle_last_assistant_message`: a trailing assistant message's
/// text is the continuation under `continue_final_message`, else a user turn.
/// The messages to render and the continuation, if the last one is such.
fn last_assistant_message(
    request: &CreateChatCompletionRequest,
    continuing: bool,
) -> Option<(minijinja::Value, String)> {
    if !matches!(
        request.messages.last(),
        Some(ChatCompletionRequestMessage::Assistant(_))
    ) {
        return None;
    }
    let mut messages = serde_json::to_value(&request.messages).ok()?;
    let list = messages.as_array_mut()?;
    let content = list.last()?["content"].as_str()?.to_owned();
    let prefix = if continuing {
        list.pop();
        content
    } else {
        *list.last_mut()? = serde_json::json!({"role": "user", "content": content});
        String::new()
    };
    Some((minijinja::Value::from_serialize(&messages), prefix))
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry, and the messages to render in place of its own.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a ChatTemplateKwargs>,
    messages: Option<minijinja::Value>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        match &self.messages {
            Some(messages) => messages.clone(),
            None => self.request.messages(),
        }
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        match self.messages {
            Some(_) => None,
            None => self.request.typed_messages(),
        }
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&ChatTemplateKwargs> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.request.mm_processor_kwargs()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use sglang_processor::ChatFormatterOptions;

    use super::*;

    /// The formatter a checkpoint with `config` and Jinja `template` selects.
    fn formatter(config: &Value, template: &str) -> ChatFormatter {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("jinja-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        let (formatter, error) = sglang_processor::select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
        std::fs::remove_dir_all(&dir).unwrap();
        formatter.unwrap_or_else(|| panic!("{error:?}"))
    }

    fn render(formatter: &ChatFormatter, body: Value) -> RenderedChat {
        let request = serde_json::from_value(body.clone()).unwrap();
        let kwargs: Option<ChatTemplateKwargs> =
            serde_json::from_value(body["chat_template_kwargs"].clone()).unwrap();
        render_chat(formatter, body, &request, kwargs.as_ref(), &HashMap::new()).unwrap()
    }

    /// Python's `_handle_last_assistant_message` on a Jinja template: the
    /// trailing assistant text is continued apart from the prompt, or asked as
    /// a user turn without `continue_final_message`.
    #[test]
    fn jinja_trailing_assistant_message_is_continued_or_asked() {
        let formatter = formatter(
            &json!({}),
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}\
             {% if add_generation_prompt %}<assistant>{% endif %}",
        );
        let messages = json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hel"}
        ]);
        let body = json!({"model": "m", "messages": messages, "continue_final_message": true});
        let rendered = render(&formatter, body);
        assert_eq!(rendered.prompt.as_str(), "<user>Hi<assistant>");
        assert_eq!(rendered.continuation, "Hel");

        let rendered = render(&formatter, json!({"model": "m", "messages": messages}));
        assert_eq!(rendered.prompt.as_str(), "<user>Hi<user>Hel<assistant>");
        assert_eq!(rendered.continuation, "");
    }

    /// The request's effort reaches a Jinja template with the thinking toggle
    /// it implies, beneath the request's own `chat_template_kwargs`.
    #[test]
    fn jinja_template_sees_the_requested_effort() {
        let formatter = formatter(
            &json!({}),
            "{{ reasoning_effort }}|{{ thinking }}|{{ enable_thinking }}",
        );
        let messages = json!([{"role": "user", "content": "Hi"}]);
        let body = json!({"model": "m", "messages": messages, "reasoning_effort": "none"});
        assert_eq!(render(&formatter, body).prompt.as_str(), "none|False|False");
        let body = json!({
            "model": "m",
            "messages": messages,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": false}
        });
        assert_eq!(render(&formatter, body).prompt.as_str(), "high|True|False");
    }

    /// A DeepSeek-V4.1 developer turn, which SGLang's encoder declines, renders
    /// through Dynamo's, as before that encoder existed.
    #[test]
    fn declined_deepseek_v41_requests_render_through_dynamo() {
        let messages = json!([
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Hi"}
        ]);
        let body = json!({"model": "m", "messages": messages});
        let formatter = ChatFormatter::DeepSeekV41;
        assert!(matches!(
            formatter.render_request(body.clone(), &HashMap::new()),
            Err(TemplateError::EngineOnly { .. })
        ));
        let prompt = render(&formatter, body).prompt.as_str().to_owned();
        assert!(
            prompt.contains("Be brief.") && prompt.contains("Hi"),
            "{prompt}"
        );
    }

    /// Qwen3.8's renderer declines floats serde_json prints unlike Python;
    /// with no engine to hand them to, the template renders them instead of
    /// the request failing.
    #[test]
    fn declined_requests_render_through_the_template() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sglang-processor/tests/fixtures/parity/qwen3.8-27b.json"
        );
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let template = fixture["chat_template"].as_str().unwrap();
        let formatter = formatter(&fixture["config"], template);
        assert!(matches!(formatter, ChatFormatter::Qwen38 { .. }));

        for name in ["history_float_arguments", "tool_float_values"] {
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == name)
                .unwrap();
            let body = case["request"].clone();
            assert!(matches!(
                formatter.render_request(body.clone(), &HashMap::new()),
                Err(TemplateError::EngineOnly { .. })
            ));
            let request: CreateChatCompletionRequest =
                serde_json::from_value(body.clone()).unwrap();
            let rendered = render_chat(&formatter, body, &request, None, &HashMap::new())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let prompt = rendered.prompt.as_str();
            assert!(
                prompt.contains("foo") && prompt.contains("<|im_start|>assistant\n<think>"),
                "{name}: {prompt}"
            );
        }
    }
}
