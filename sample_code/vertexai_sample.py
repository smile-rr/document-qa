from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="shining-axis-449814-p0",
      location="us-central1",
  )


  model = "gemini-2.0-flash-lite-preview-02-05"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text="""豺狼的日子讲了什么故事""")
      ]
    ),
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    # safety_settings = [types.SafetySetting(
    #   category="HARM_CATEGORY_HATE_SPEECH",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_DANGEROUS_CONTENT",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_HARASSMENT",
    #   threshold="OFF"
    # )],
    response_mime_type = "application/json",
    response_schema = {"type":"OBJECT","properties":{"response":{"type":"STRING"}}},
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()