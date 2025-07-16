# gpt_stream.py  ——  reads llm_queue, streams GPT tokens to tts_queue
import asyncio, os, openai, sys

openai.api_key = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY not set")

llm_queue  = None    # we'll inject these two from main.py
tts_queue  = None

async def llm_loop():
    while True:
        prompt = await llm_queue.get()          # wait for ASR
        reply  = ""

        stream = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",                # or gpt-3.5-turbo
            stream=True,
            messages=[
                {"role": "system",
                 "content": "You are a friendly AI avatar interviewer."},
                {"role": "user", "content": prompt},
            ],
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            print(delta, end="", flush=True)    # live console feedback
            await tts_queue.put(delta)          # hand off to TTS

        await tts_queue.put("\n<eos>")          # signal sentence done
