#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
from typing import Optional
import gradio as gr
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def pull_messages_from_step(step_log: MemoryStep):
    """Extract ChatMessage objects from agent steps with proper nesting."""
    if isinstance(step_log, ActionStep):
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = step_log.model_output.strip()
            model_output = re.sub(r"\s*<end_code>", " ", model_output)
            yield gr.ChatMessage(role="assistant", content=model_output)

        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            for tool_call in step_log.tool_calls:
                used_code = tool_call.name == "python_interpreter"
                parent_id = f"call_{len(step_log.tool_calls)}"
                args = tool_call.arguments
                content = str(args.get("answer", str(args))) if isinstance(args, dict) else str(args).strip()

                if used_code:
                    content = re.sub(r".*?\n", "", content)
                    content = re.sub(r"\s*<end_code>\s*", "", content).strip()
                    if not content.startswith("python"):
                        content = f"python\n{content}\n"
                
                yield gr.ChatMessage(
                    role="assistant",
                    content=content,
                    metadata={"title": f"🛠️ Used tool {tool_call.name}", "id": parent_id, "status": "pending"},
                )

                if hasattr(step_log, "observations") and step_log.observations:
                    log_content = step_log.observations.strip()
                    if log_content:
                        yield gr.ChatMessage(
                            role="assistant",
                            content=f"{log_content}",
                            metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                        )
                
                if hasattr(step_log, "error") and step_log.error:
                    yield gr.ChatMessage(
                        role="assistant",
                        content=str(step_log.error),
                        metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                    )
                
                #tool_call.metadata["status"] = "done"

        elif hasattr(step_log, "error") and step_log.error:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            step_footnote += f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else ""
            step_footnote += step_duration
        step_footnote = f"<span style='color: #bbbbc2; font-size: 12px;'>{step_footnote}</span> "
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")




def stream_to_gradio(agent, task, reset_agent_memory=False, additional_args=None):
    """Runs an agent with a given task and streams messages from the agent as Gradio ChatMessages."""
    if not isinstance(task, dict):
        task = {"text": str(task).strip(), "image": None}

    text_input = task.get("text", "").strip()
    file_path = task.get("file", None)

    # If an image is provided, explicitly call the object detection tool
    if file_path:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in [".png", ".jpg", ".jpeg"]:
            text_input += f"\n\n**Image Provided:** {file_path}"
        elif file_ext in [".pdf", ".txt", ".docx"]:
            text_input += f"\n\n**File Provided:** {file_path}"
        

    for step_log in agent.run(text_input, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        for message in pull_messages_from_step(step_log):
            yield message
    
    final_answer = handle_agent_output_types(step_log)
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


class GradioUI:
    def __init__(self, agent, file_upload_folder: str = "uploaded_files"):
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        os.makedirs(self.file_upload_folder, exist_ok=True)

    def interact_with_agent(self, text_input, file_input, messages):
        input_data = {"text": text_input, "file": None}

        if file_input:
            file_path = file_input.name  # Use the direct file path
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Only move if necessary (avoid SameFileError)
            new_file_path = os.path.join(self.file_upload_folder, os.path.basename(file_path))

            if os.path.exists(file_path):
                shutil.copy(file_path, new_file_path) 
            if file_path != new_file_path:
                shutil.move(file_path, new_file_path)  # Move instead of copy
            
            input_data["file"] = new_file_path

        messages.append(gr.ChatMessage(role="user", content=f"**Text:** {text_input}\n**File Provided:** {bool(file_input)}"))
        yield messages

        for msg in stream_to_gradio(self.agent, task=input_data, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages
        
    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )
    
    def launch(self, **kwargs):
        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(label="Agent", type="messages")
            # if self.file_upload_folder is not None:
            #     upload_file = gr.File(label="Upload a file")
            #     upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
            #     upload_file.change(
            #         self.upload_file,
            #         [upload_file, file_uploads_log],
            #         [upload_status, file_uploads_log],
            #     )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            #image_input = gr.Image(type="pil", label="Upload an Image (Optional)")
            upload_file = gr.File(label="Upload a File (Image, PDF, TXT, DOC)", type="filepath", file_types=[".png", ".jpg", ".jpeg", ".pdf", ".txt", ".docx"])
            submit_button = gr.Button("Send")
            submit_button.click(self.interact_with_agent, [text_input, upload_file, stored_messages], [chatbot])
            
            
        demo.launch(debug=True, share=True, **kwargs)

