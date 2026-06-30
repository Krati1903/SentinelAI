# SentinelAI – AI Interview Intelligence Platform

> An AI-powered interview monitoring and recruiter intelligence platform that combines Computer Vision, Large Language Models, Context-Augmented Generation (CAG), and Retrieval-Augmented Generation (RAG) to assist recruiters in conducting secure online interviews and assessments.

---

## Overview

SentinelAI is an intelligent interview monitoring platform designed to improve the integrity and efficiency of online interviews.

The system continuously analyzes candidate behavior using Computer Vision techniques and generates structured interview insights for recruiters using Large Language Models.

Unlike conventional proctoring systems that only flag suspicious activities, SentinelAI transforms interview session data into actionable recruiter insights through AI-powered agents.

---

## Key Features

### Real-Time Interview Monitoring

- Face Detection
- Eye Gaze Tracking
- Head Pose Estimation
- Multiple Person Detection
- Object Detection (YOLOv8)
- Rule-based Cheating Detection
- Dynamic Risk Scoring
- Automated Session Logging

---

### AI Recruiter Summary Agent (CAG)

The Recruiter Summary Agent follows a **Context-Augmented Generation (CAG)** approach.

Instead of sending raw interview logs directly to an LLM, SentinelAI first analyzes the session logs to extract meaningful interview insights. This structured context is then provided to the LLM to generate recruiter-friendly summaries.

Generated Summary includes:

- Interview Overview
- Behaviour Analysis
- Risk Assessment
- Key Evidence
- Recruiter Recommendation
- Confidence Score

---

### AI Recruiter Assistant (RAG)

The Recruiter Assistant is a **RAG-based conversational assistant** built for recruiters.

It retrieves relevant interview summaries, session logs and company policy documents before answering recruiter queries.

Example questions:

- Why was this candidate flagged?
- Explain the final risk score.
- Show suspicious events.
- Did multiple people appear?
- Summarize candidate behaviour.
- Should this interview be reviewed manually?

---

## System Architecture

```
                    Candidate

                        │

                        ▼

              Webcam Permission

                        │

                        ▼

               FastAPI Backend

                        │

                        ▼

         Computer Vision Pipeline

        ┌────────────────────────────┐
        │ OpenCV                     │
        │ MediaPipe                  │
        │ YOLOv8                     │
        └────────────────────────────┘

                        │

                        ▼

            Behaviour Detection

        • Face Detection
        • Eye Gaze
        • Head Pose
        • Multiple Persons
        • Objects
        • Rule Engine

                        │

                        ▼

              Dynamic Risk Engine

                        │

                        ▼

               Session Logs

          ┌─────────────────────┐
          │ Summary Agent (CAG)  │
          └─────────────────────┘

                        │

                        ▼

          AI Interview Summary

                        │

          ┌─────────────────────┐
          │ Recruiter Assistant │
          │       (RAG)         │
          └─────────────────────┘

                        │

                        ▼

             Recruiter Dashboard
```

---

# Tech Stack

## Backend

- FastAPI
- Python

## Computer Vision

- OpenCV
- MediaPipe
- YOLOv8

## AI / LLM

- LangChain
- Groq LLM

## AI Architectures

- Context-Augmented Generation (CAG)
- Retrieval-Augmented Generation (RAG)

## Database

- MongoDB / Firebase *(depending on deployment)*

---

# AI Pipeline

## Step 1

Candidate joins interview.

↓

## Step 2

Frames are streamed to the FastAPI backend.

↓

## Step 3

Computer Vision analyzes every frame.

Detected signals include

- Face Count
- Person Count
- Eye Gaze
- Head Pose
- Objects
- Risk Score
- Warnings

↓

## Step 4

Session logs are generated.

↓

## Step 5

The CAG Summary Agent analyzes the logs and generates an AI-powered recruiter summary.

↓

## Step 6

The RAG Recruiter Assistant answers recruiter questions using interview summaries, session logs and company policies.

---

# Project Structure

```
SentinelAI

├── app
│   ├── api
│   ├── agents
│   ├── detectors
│   ├── models
│   ├── prompts
│   ├── reports
│   ├── knowledge
│   ├── pipeline.py
│   ├── risk_engine.py
│   ├── logger.py
│   ├── session_manager.py
│   └── camera.py
│
├── logs
├── session_logs
├── requirements.txt
├── main.py
└── README.md
```

---

# Current Capabilities

✅ Real-time interview monitoring

✅ Dynamic risk scoring

✅ Session logging

✅ Computer Vision pipeline

✅ AI Recruiter Summary Agent (CAG)

✅ AI Recruiter Assistant (RAG)

---

# Future Improvements

- Real-time recruiter dashboard using WebSockets
- Multi-camera support
- Voice activity detection
- Speaker verification
- Face recognition
- Candidate authentication
- PDF interview reports
- Analytics dashboard
- Cloud deployment
- Multi-language interview support

---

# Why SentinelAI?

Traditional online proctoring systems only detect suspicious activities.

SentinelAI goes one step further by combining Computer Vision with Large Language Models to transform interview session data into structured recruiter insights, making online interviews more secure, explainable and efficient.
