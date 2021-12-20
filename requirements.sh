#!/bin/bash

# 1. Crawling
sh crawling/requirements.sh

# 2. Serving
## 2.1 Fastapi
sh serving/requirements.sh
## 2.2 Airflow
sh airflow/requirements.sh

# 3. Clustering
sh clustering/requirements.sh

# 4. Summarization Model
sh summary/requirements.sh

# 5. TTS
sh tts/requirements.sh


