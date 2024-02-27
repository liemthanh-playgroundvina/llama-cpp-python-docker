# Build and start
build:
	docker build -t open-llm -f Dockerfile .
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
#
#	huggingface-cli download lmstudio-ai/gemma-2b-it-GGUF gemma-2b-it-q4_k_m.gguf --local-dir ./models --local-dir-use-symlinks False

download_model_alpine:
	curl -o ./models/gemma-2b-it-q4_k_m.gguf https://aiservices-bucket/ai_model/GGUF/gemma-2b-it-q4_k_m.gguf
