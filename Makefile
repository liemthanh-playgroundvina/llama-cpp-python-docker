# Build and start
build:
	docker build -t endpoints-fastapi -f endpoints/Dockerfile endpoints
	#docker build -t open-llm-python -f open-llm-python/Dockerfile open-llm-python
	docker build -t open-llm-cpp -f open-llm-cpp/Dockerfile open-llm-cpp
pull:
	#docker pull liemthanh/open-llm-python:latest
	docker pull liemthanh/open-llm-cpp:latest
	docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.1

start:
	docker compose -f docker-compose.yml down
	docker compose -f docker-compose.yml up -d

stop:
	docker compose -f docker-compose.yml down

#upload_model:
#
#	https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/blob/main/gemma-2b-it-q4_k_m.gguf
#
#	aws s3 cp D:\PlayGround\llama-cpp-python-docker\models\gemma-2b-it-q4_k_m.gguf s3://aiservices-bucket/ai_model/GGUF/
#	aws s3 cp D:\PlayGround\llama-cpp-python-docker\models\qwen1_5-0_5b-chat-q4_k_m.gguf s3://aiservices-bucket/ai_model/GGUF/
#
#	huggingface-cli download lmstudio-ai/gemma-2b-it-GGUF gemma-2b-it-q4_k_m.gguf --local-dir ./models --local-dir-use-symlinks False

download_model:
	curl -o ./models/GGUF/gemma-2b-it-q4_k_m.gguf https://aiservices-bucket.s3.amazonaws.com/ai_model/GGUF/gemma-2b-it-q4_k_m.gguf
	curl -o ./models/GGUF/qwen1_5-0_5b-chat-q4_k_m.gguf https://aiservices-bucket.s3.amazonaws.com/ai_model/GGUF/qwen1_5-0_5b-chat-q4_k_m.gguf
