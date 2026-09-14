# 03 · Vision & Multimodal Inputs

Everything so far has been text in, text out. Modern LLMs also accept
**images and PDFs** directly in the message content, letting you build
document Q&A, chart reading, screenshot debugging, and receipt/form
extraction without a separate OCR pipeline.

## Sending an image

Images go in as base64-encoded content blocks alongside text, in the same
`messages` structure you already know:

```python
from dotenv import load_dotenv
import anthropic, base64, httpx

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

def image_block(path: str, media_type: str = "image/png") -> dict:
    data = base64.standard_b64encode(open(path, "rb").read()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }

resp = client.messages.create(
    model=MODEL,
    max_tokens=400,
    messages=[{
        "role": "user",
        "content": [
            image_block("dashboard_screenshot.png"),
            {"type": "text", "text": "What's the error shown in this screenshot, "
                                      "and which line number does it reference?"},
        ],
    }],
)
print(resp.content[0].text)
```

Put the image block(s) before the text instruction — the model reads
content blocks in order, and grounding the instruction *after* the image it
refers to reduces ambiguity, especially with multiple images.

Images can also be a direct URL instead of base64, when the model is
allowed to fetch it:

```python
{"type": "image", "source": {"type": "url", "url": "https://example.com/chart.png"}}
```

## Multiple images and comparison tasks

Label images explicitly in your text when comparing them — the model sees
blocks in sequence, not named variables:

```python
resp = client.messages.create(
    model=MODEL, max_tokens=400,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image 1 (before):"},
            image_block("before.png"),
            {"type": "text", "text": "Image 2 (after):"},
            image_block("after.png"),
            {"type": "text", "text": "List every visible difference between Image 1 and Image 2."},
        ],
    }],
)
```

## Document Q&A with PDFs

PDFs are sent the same way, as a `document` content block. The model
processes both the extracted text *and* the page layout/visuals — useful
for tables, forms, and scanned pages where plain text extraction loses
structure:

```python
def pdf_block(path: str) -> dict:
    data = base64.standard_b64encode(open(path, "rb").read()).decode("utf-8")
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }

resp = client.messages.create(
    model=MODEL, max_tokens=600,
    messages=[{
        "role": "user",
        "content": [
            pdf_block("invoice_q3.pdf"),
            {"type": "text", "text": "Extract every line item as a table: "
                                      "description, quantity, unit price, total."},
        ],
    }],
)
print(resp.content[0].text)
```

Combine this with module 4's structured output (a JSON schema for line
items) when you need the extraction to feed a downstream system rather than
a human reader.

## Multimodal prompt design

A few things behave differently from pure-text prompting:

- **Resolution matters, but bigger isn't always better.** Very large images
  are downscaled internally; a full 4K screenshot of a tiny error message
  may render the text unreadable to the model even though it looks fine to
  you. Crop to the relevant region when precision matters.
- **Ask the model to describe before it answers**, for anything requiring
  precise reading (numbers, small text): "First transcribe every number you
  see in the chart, then answer the question." This surfaces
  misreadings you can catch before they propagate into a wrong answer.
- **Token cost scales with image size and count** — an image roughly
  contributes a token cost similar to a few hundred to over a thousand
  words of text depending on resolution, so a document Q&A pipeline over
  hundreds of pages needs the same cost tracking as Level 1's module 8.

## How It Actually Works

Vision-language models are trained to map image pixels into the *same*
embedding space that text tokens live in. Concretely, an image is split
into a grid of patches; a vision encoder (typically a separately-trained
component) converts each patch into a vector, and those vectors are
projected into the transformer's token embedding dimensionality so they can
sit in the same input sequence as text token embeddings — architecturally,
an image becomes a run of extra "tokens" the transformer attends over
exactly like word tokens, which is also why larger or more numerous images
consume more of the context window and cost more.

This is why order and framing in your prompt matter: attention has no
built-in notion of "this image is what I'm asking about" — it only sees a
flat sequence of image-patch-tokens and text-tokens, and relies on learned
associations (reinforced by training data where instructions typically
follow the image or document they refer to) to connect a question to the
right visual tokens. It's also why asking the model to transcribe fine
detail first genuinely helps: transcription forces an explicit intermediate
step where the visual-to-text mapping is committed to the output as text
tokens, which then become part of what subsequent tokens (the actual
answer) can condition on — the same "let the model show its work so later
tokens can use it" mechanism that makes chain-of-thought help on text-only
reasoning.

## Cheat sheet

| Task | Content block |
|---|---|
| Photo, screenshot, chart | `{"type": "image", "source": {...}}` |
| PDF (text + layout) | `{"type": "document", "source": {...}}` |
| Multiple images | Multiple blocks in one message, in reading order |
| Precision on small text | Ask for transcription first, then the answer |
| Cost | Images/PDFs consume real context tokens — budget for them |

## Exercise

Take three receipt images (or generate mock ones) and build a pipeline that
sends each to the model with a JSON schema (from module 4) requesting
`{merchant, date, total, line_items: [...]}`, then sums `total` across all
three. Deliberately test with one blurry or rotated image and observe how
the extraction degrades — add a confidence field to your schema and have
the model flag low-confidence extractions for human review instead of
silently guessing.
