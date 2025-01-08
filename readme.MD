# Social Media Performance Analysis

This repository contains the implementation of the **Social Media Performance Analysis** project, developed for the **Level Supermind Hackathon** held in Mumbai (offline). The objective was to build a basic analytics module using **Langflow** and **DataStax Astra DB** to analyze engagement data from mock social media accounts.

---

## 🚀 Project Objective

Build an analytics module to:
1. Fetch and store simulated social media engagement data.
2. Analyze post performance based on engagement metrics.
3. Provide insights using GPT integration.

---

## 🛠 Tools & Technologies Used

- **Langflow:** For creating workflows and integrating GPT.
- **DataStax Astra DB:** For database operations to store and query engagement data.

---

## 📋 Task Details

### 1. **Fetch Engagement Data**
- Created a mock dataset simulating social media engagement (e.g., likes, shares, comments, post types like carousel, reels, static images).
- Stored the dataset in **DataStax Astra DB**.

### 2. **Analyze Post Performance**
- Built a Langflow workflow to:
  - Accept post types (carousel, reels, static images) as input.
  - Query the dataset in Astra DB to calculate average engagement metrics for each post type.

### 3. **Provide Insights**
- Used GPT integration in Langflow to generate insights based on engagement data.
- Example outputs:
  - "Carousel posts have 20% higher engagement than static posts."
  - "Reels drive 2x more comments compared to other formats."

---

## 📝 Submission Requirements Fulfilled

1. **Project Repository:** Linked to **Level Supermind Hackathon**.
2. **Demo Video:** [YouTube Video Link](#) _(Update with your video link)_.
   - Describes the Langflow workflow, DataStax Astra DB usage, and GPT integration.
3. **Public Repository:** All project files are available in this public repository.
4. **Well-written Description:** This README provides a clear overview of the project.
5. Open Langflow and load the `workflow.json` file to view and execute the workflow.

---

## 🛠 Setup Instructions

### Prerequisites
- Python 3.8 or later installed
- Node.js (for Langflow UI)
- DataStax Astra account (free tier available)

### 1. Clone the Repository
```bash
git clone https://github.com/ShreyasGandhi0607/social_media_performace_analysis
cd social-media-performance-analysis
```

### 2. Set Up DataStax Astra DB
1. Sign up for a free DataStax Astra account at [Astra DB](https://www.datastax.com/astra).
2. Create a new database (Keyspace: `social_media_data`).
3. Download the database secure connect bundle and place it in the project root directory.

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Load Langflow and Import Workflow
```bash
pip install langflow
langflow
```
1. Access Langflow at `http://localhost:7860`.
2. Import `workflow.json` from the project directory.
3. Configure nodes to connect to your Astra DB by uploading the secure connect bundle.


### 5. Run Analysis
1. Launch Langflow.
2. Select the imported workflow and click "Run".
3. Input post types (carousel, reels, etc.) to analyze engagement metrics.

---

## ⚙️ Troubleshooting
- **Langflow not launching?** Ensure Python and Node.js are properly installed.
- **Astra DB connection errors?** Double-check the secure connect bundle and re-upload it in Langflow.
- **Missing dependencies?** Re-run `pip install -r requirements.txt`.
