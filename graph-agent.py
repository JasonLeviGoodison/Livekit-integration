import asyncio
from datetime import datetime

from aiofile import async_open as open
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

load_dotenv(dotenv_path=".env.local")

async def entrypoint(ctx: JobContext):
    # Define system prompts for different states
    introduction_prompt = (
        "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
        "Use short and concise responses, and avoid unpronounceable punctuation. "
        "If asked about your state, respond with 'Introduction'."
    )
    end_prompt = (
        "You have reached the end state. Thank you for using LiveKit's voice assistant. "
        "If asked about your state, respond with 'End'."
    )

    # Define the possible states
    STATES = {
        "introduction": introduction_prompt,
        "end": end_prompt
    }

    # Start in the introduction state
    current_state = "introduction"

    async def update_chat_context(state: str):
        system_prompt = STATES.get(state)
        if not system_prompt:
            raise ValueError(f"Undefined state: {state}")
        return llm.ChatContext().append(role="system", text=system_prompt)

    initial_ctx = await update_chat_context(current_state)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    def custom_before_llm_cb(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext) -> llm.LLMStream:
        # Custom logic before calling the LLM
        print("Custom before LLM callback invoked", chat_ctx.messages[-1])
        user_text = chat_ctx.messages[-1].content.strip().lower() if chat_ctx.messages else ""
        
        nonlocal current_state
        if "next" in user_text:
            if current_state == "introduction":
                current_state = "end"
                new_system_prompt = STATES[current_state]
                # Directly modify agent's chat context
                agent.chat_ctx.append(role="system", text=new_system_prompt)
                
                asyncio.create_task(agent.say(
                    "You have moved to the end state. How can I assist you further?",
                    allow_interruptions=True
                ))
            elif current_state == "end":
                asyncio.create_task(agent.say(
                    "You are already in the end state.",
                    allow_interruptions=True
                ))
        
        # Use the agent's updated chat context
        return agent.llm.chat(chat_ctx=agent.chat_ctx)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        before_llm_cb=custom_before_llm_cb
    )

    log_queue = asyncio.Queue()

    agent.start(ctx.room)

    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        print("HERHERHEREHRHERHERHERHERHEHERHERHR")
        user_text = msg.content.strip().lower()
        print("User text:" + user_text)

        # Update chat context with user's message
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.append(role="user", text=msg.content)
        # The response generation is now handled by the custom_before_llm_cb

        # Logging the committed user speech
        if isinstance(msg.content, list):
            content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else x for x in msg.content
            )
        else:
            content = msg.content
        log_queue.put_nowait(f"[{datetime.now()}] USER:\n{content}\n\n")

    @agent.on("agent_speech_committed")
    async def on_agent_speech_committed(msg: llm.ChatMessage):
        print("ON AGENT MESSGE HANDLER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log_queue.put_nowait(f"[{datetime.now()}] AGENT:\n{msg.content}\n\n")

    async def write_transcription():
        async with open("transcriptions.log", "a") as f:
            while True:
                msg = await log_queue.get()
                if msg is None:
                    break
                await f.write(msg)

    write_task = asyncio.create_task(write_transcription())

    async def finish_queue():
        log_queue.put_nowait(None)
        await write_task

    ctx.add_shutdown_callback(finish_queue)

    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))