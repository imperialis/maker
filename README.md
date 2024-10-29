# Video to GIF Creation App

## Project Purpose
The **Video to GIF Creation App** is designed to allow users to upload videos and search for specific moments within those videos based on text queries. Using advanced machine learning techniques, the app processes the video to generate embeddings that help identify the most relevant clips. Users can then retrieve GIFs of these clips directly from their uploaded videos, making it easier to share memorable moments.

## Key Features
- **Text-Based Search**: Input a query and retrieve relevant moments from the uploaded videos.
- **GIF Generation**: Automatically create GIFs from the identified video segments.
- **ChromaDB Integration**: Efficiently manage and store video embeddings to enable fast search capabilities.

## How It Works
1. **Video Upload**: Users upload one or more videos.
2. **Frame Extraction**: The app extracts frames from the videos at specified intervals.
3. **Embedding Generation**: Using a pre-trained CLIP model, the app generates embeddings for each frame and stores them in ChromaDB.
4. **Query Input**: Users input a text query describing what they are looking for.
5. **Similarity Search**: The app calculates cosine similarity between the query embedding and video embeddings to find the best match.
6. **GIF Creation**: The app creates and displays GIFs of the relevant video segments.

## Installation Guide

### Prerequisites
- Python 3.7 or higher
- `pip` for installing Python packages

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/imperialis/maker.git
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.**Install Dependencies**
   - pip install -r requirements.txt

4.**Set Up Video Directory**
   - Create a videos Directory and place your video file in the videos/ directory. You can modify the video_paths list in the script to include the paths to your videos.
   
5. **Run the Application**
   - python gifmaker.py
   
7. **Access the App**
   - Open your web browser and go to http://localhost:7860 to interact with the app.
   
**Dependencies**
- Torch: For running the machine learning model.
- Transformers: For accessing the CLIP model.
- Gradio: For creating the web interface.
- ChromaDB: For managing video embeddings.
- MoviePy: For video processing tasks.

**Example Queries**
- You can try using the following queries to retrieve GIFs from your videos:

- "A funny moment"
- "A cute animal"
- "An exciting event"

**Sample Output**
![image](https://github.com/user-attachments/assets/2ddedb82-4ec8-411d-b5b9-0d20fffbebb3)


**Contribution**
- Feel free to fork the repository and submit pull requests. Contributions are welcome!

**License**
- This project is licensed under the MIT License. See the LICENSE file for details.
