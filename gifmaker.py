# TODO#1: Import necessary libraries
import os
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
import moviepy.editor as mpy

# TODO#2: Setup ChromaDB
client = chromadb.Client()

# Create a new collection for storing video embeddings
collection = client.create_collection("video_collection")

# TODO#3: Load CLIP model and processor for generating image and text embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# TODO#4: Load and preprocess videos
video_paths = [
    "videos/tezt.mp4",  # add a sample video
    # Add more video paths as needed
]
gif_dir = '/workspace/MAKER/chromadb-gifmaker/gifs'
# Check if the directory exists, if not, create it
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

# Preprocess videos and generate embeddings
video_embeddings = []
frame_paths = []  # To store paths of frames for later GIF creation

for video_path in video_paths:
    clip = mpy.VideoFileClip(video_path)
    duration = clip.duration

    # Extract frames every 2 seconds
    for t in range(0, int(duration), 2):
        frame = clip.get_frame(t)
        frame_path = f"{gif_dir}/frame_{len(video_embeddings)}.jpg"
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)
        
        # Generate embedding for the frame
        inputs = processor(images=[Image.fromarray(frame)], return_tensors="pt", padding=True)

        with torch.no_grad():
            video_embedding = model.get_image_features(**inputs).numpy()
            video_embeddings.append(video_embedding.flatten().tolist())  # Flatten and convert to list

# TODO#5: Add video embeddings to the collection with metadata
collection.add(
    embeddings=video_embeddings,
    metadatas=[{"video": video_path} for video_path in video_paths for _ in range(len(video_embeddings) // len(video_paths))],
    ids=[str(i) for i in range(len(video_embeddings))]
)

# TODO#6: Create a function to calculate accuracy score based on cosine similarity
def calculate_accuracy(video_embedding, query_embedding):
    # Cosine similarity between query and video embeddings
    similarity = cosine_similarity([video_embedding], [query_embedding])[0][0]
    return similarity

# Define Gradio function
def search_video(query):
    if not query.strip():
        return None, "Oops! You forgot to type something in the query input!", ""

    print(f"\nQuery: {query}")

    # Start measuring the query processing time
    start_time = time.time()

    # TODO#7: Generate an embedding for the query text
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy().flatten()

    # TODO#8: Perform a vector search in the collection
    similarities = [calculate_accuracy(video_embedding, query_embedding) for video_embedding in video_embeddings]
    best_index = similarities.index(max(similarities))

    # TODO#9: Retrieve the best frame for the matched query
    best_frame_path = frame_paths[best_index]
    
    # Calculate accuracy score
    accuracy_score = max(similarities)

    # End time for query processing
    end_time = time.time()
    query_time = end_time - start_time

    # TODO#10: Provide option to create a GIF from the best frame
    gif_path = f"{gif_dir}/output_{best_index}.gif"
    clip = mpy.VideoFileClip(video_paths[0])  # Assuming you only process one video
    clip.subclip(best_index * 2, best_index * 2 + 2).write_gif(gif_path)  # Create a GIF from the relevant segment

    return gif_path, f"Accuracy score: {accuracy_score:.4f}\nQuery time: {query_time:.4f} seconds", best_frame_path

# Suggested queries
queries = [
    "A funny moment",
    "A cute animal",
    "An exciting event",
    # Add more queries as needed
]

# Gradio Interface Layout
with gr.Blocks() as gr_interface:
    gr.Markdown("# Video to Sticker/GIF Creation App")
    with gr.Row():
        # Left Panel
        with gr.Column():
            gr.Markdown("### Input Panel")
            custom_query = gr.Textbox(placeholder="Enter your custom query here", label="What are you looking for?")
            submit_button = gr.Button("Submit Query")
            cancel_button = gr.Button("Cancel")

            gr.Markdown("#### Suggested Search Phrases")
            with gr.Row(elem_id="button-container"):
                for query in queries:
                    gr.Button(query).click(fn=lambda q=query: q, outputs=custom_query)

        # Right Panel
        with gr.Column():
            gr.Markdown("### Retrieved GIF")
            gif_output = gr.Image(label="Result GIF")
            accuracy_output = gr.Textbox(label="Performance")

        # Button click handler for custom query submission
        submit_button.click(fn=search_video, inputs=custom_query, outputs=[gif_output, accuracy_output])
        cancel_button.click(fn=lambda: (None, ""), outputs=[gif_output, accuracy_output])

# TODO#14: Launch the Gradio interface
gr_interface.launch()
