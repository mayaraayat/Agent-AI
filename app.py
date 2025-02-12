from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import torch
from Gradio_UI import GradioUI
from PIL import Image

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_cutom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool 
def generate_images(prompt:str)-> Image.Image:
    """A tool that generates images from text prompts
    Args:
        prompt: A string representing the text prompt
    """
    # Import tool from Hub
    image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True) 
    try:
        # Generate image from prompt
        image = image_generation_tool(prompt)
        return image
    except Exception as e:
        return f"Error generating image from prompt '{prompt}': {str(e)}"


@tool
def detect_objects_in_image(image_path: str) -> str:
    """Detects objects in an image using a pre-trained model.
    
    Args:
        image_path: Path to the image file.
    """
    from transformers import pipeline

    # Load pre-trained object detection model
    model = pipeline("object-detection")
    
    # Load image
    image = Image.open(image_path)

    # Detect objects in image
    results = model(image)
        # Return list of detected
    return ", ".join(set([obj["label"] for obj in results]))

@tool 
def generate_image_caption(image_path: str) -> str:
    """Generates a caption describing the contents of an image.
    
    Args:
        image_path: Path to the image file.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        caption = model.generate(**inputs)
    
    caption = processor.decode(caption[0], skip_special_tokens=True)
    return caption

@tool
def summarize_text(text: str) -> str:
    """Summarizes long text into key points.
    
    Args:
        text: The text to summarize.
    """
    from transformers import pipeline
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

@tool 
def detect_ner(text:str) -> str:
    """Detects named entities in text.
    
    Args:
        text: The text to analyze.
    """
    
    from transformers import pipeline

    # Load the pre-trained NER model
    ner_model = pipeline("ner", grouped_entities=True)

    # Process the text and extract entities
    entities = ner_model(text)

    # Structure the extracted entities in a dictionary
    entity_dict = {}
    for entity in entities:
        entity_type = entity["entity_group"]
        entity_name = entity["word"]

        if entity_type not in entity_dict:
            entity_dict[entity_type] = []
        
        entity_dict[entity_type].append(entity_name)

    return entity_dict

@tool
def read_files(file_path:str) -> str:
    """Reads the contents of a pdf, txt and doc files and returns the text.
    
    Args:
        file_path: The path to the file to read.
    """
    
    import PyPDF2
    import docx
    import textract
    
    if file_path.endswith('.pdf'):
        pdfFileObj = open(file_path, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        text = ""
        for page_num in range(num_pages):
            pageObj = pdfReader.pages[page_num]
            text += pageObj.extract_text()
        pdfFileObj.close()
    
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text
    
    elif file_path.endswith('.txt'):
        text = textract.process(file_path).decode()
    
    return text
    
    

final_answer = FinalAnswerTool()
model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id= 'Qwen/Qwen2.5-Coder-32B-Instruct', #'https://wxknx1kg971u7k1n.us-east-1.aws.endpoints.huggingface.cloud',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[read_files,detect_ner,summarize_text, generate_image_caption, detect_objects_in_image,DuckDuckGoSearchTool(), generate_images, get_current_time_in_timezone, final_answer], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()