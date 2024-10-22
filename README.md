# Music Recommendation System - DS 340 Project

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://uobevents.eventsair.com/cikm2023//" alt="Conference">
        <img src="https://img.shields.io/badge/CIKM'23-green" /></a>
</p>

The official source code for "[MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement](https://arxiv.org/abs/2308.09649)", accepted at CIKM 2023.


## Overview

Overview
This project aims to develop an advanced Music Recommendation System by combining methodologies from three influential research papers: MUSE, Learning Content Similarity for Music Recommendation, and Why People Skip Music? On Predicting Music Skips using Deep Reinforcement Learning. Our goal is to create a more robust recommendation system that effectively addresses the challenges of shuffle play sessions, the cold-start problem, and song skip prediction.

This Github is based off the original github found here, and includes our changes and improvements to this project: https://github.com/yunhak0/MUSE/blob/main/README.md 

# Research Papers
1. MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement
Authors: Yunhak Oh, Sukwon Yun, Dongmin Hyun, Sein Kim, Chanyoung Park
Summary: The MUSE framework introduces a self-supervised algorithm that enhances music recommendations during shuffle play sessions. It tackles the noise introduced by random song selections and employs transition-based augmentation and fine-grained matching strategies.

2. Learning Content Similarity for Music Recommendation
Authors: Brian McFee, Gert Lanckriet, Luke Barrington
Summary: This paper addresses the cold-start problem in recommendation systems by proposing a hybrid approach that integrates collaborative filtering (CF) data with content-based audio similarity. It utilizes Vector Quantization (VQ) of audio features to improve recommendation accuracy for lesser-known tracks.

3. Why People Skip Music? On Predicting Music Skips using Deep Reinforcement Learning
Authors: Francesco Meggetto, Crawford Revie, John Levine, Yashar Moshfeghi
Summary: This research investigates user skipping behavior on music streaming platforms using Deep Reinforcement Learning (DRL). By analyzing large-scale user interaction data, the study develops a model to predict when and why users skip songs, leading to more effective and personalized recommendations.


# Data Sources
To build and evaluate our music recommendation system, we utilize a variety of data sources that reflect real-world user interactions and song attributes:

Spotify Listening Sessions Dataset:

This large-scale dataset includes millions of user interactions on Spotify, capturing both shuffle and non-shuffle play sessions. It provides detailed information about the songs played, user preferences, and session metadata, allowing us to train and validate the MUSE framework effectively.

# User Interaction Data:

We analyze large volumes of user interaction data that track song skips, listens, and user ratings across different platforms. This data includes:
- Song Attributes: Features such as genre, tempo, artist popularity, and release year.
- User Preferences: Information derived from user profiles, listening history, and feedback.
- Skip Behavior Data: Instances of song skips, including timestamps and contextual factors like time spent on each song.

Audio Feature Datasets:

We extract low-level audio features (e.g., Mel-frequency cepstral coefficients, spectral contrast) from songs to enhance content-based similarity assessments. These features help us understand the acoustic characteristics of songs and how they relate to user preferences.

# Methodology
Our system integrates the methodologies from the three papers to create a comprehensive recommendation engine:

- Shuffle Play Handling: Implementing the MUSE framework, we utilize self-supervised learning to improve recommendations during shuffle play sessions by mitigating the noise caused by random track transitions.
- Cold-Start Problem Solution: We employ hybrid techniques combining collaborative filtering with content-based approaches to ensure accurate recommendations for new and less popular songs.
- Skip Prediction: By applying deep reinforcement learning, we predict user skipping behavior, adjusting recommendations in real-time to align with user preferences and enhance satisfaction.

# Results
Preliminary results indicate significant improvements in recommendation accuracy and user satisfaction:

Our system demonstrates a substantial reduction in the skip rate by anticipating user preferences.
MUSE's self-supervised model outperforms 12 baseline models in shuffle play environments, showcasing its robustness in real-world scenarios.

# Future Work
To further enhance our music recommendation system, we plan to:

Expand Data Sources: Incorporate additional datasets from other music streaming platforms to increase the diversity of user interactions and song features.
User Feedback Mechanism: Implement a mechanism for real-time user feedback to continually refine the recommendation engine based on actual user experiences.
Performance Evaluation: Conduct extensive A/B testing to compare our system against existing recommendation algorithms across various user demographics.

# Conclusion
By synthesizing insights from the MUSE framework, content similarity learning, and skip prediction models, we are creating a music recommendation system that not only addresses the unique challenges posed by shuffle play but also improves overall user satisfaction and engagement.

<p float="middle">
  <img src="https://github.com/yunhak0/MUSE/assets/40286691/5eaf7132-9df0-4ff8-820c-88fc646c1f62" width="51.5%" /><img src="https://github.com/yunhak0/MUSE/assets/40286691/a6296eca-cd3c-4741-b069-7dbaac535bee" width="40%" /> 
</p>
