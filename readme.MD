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