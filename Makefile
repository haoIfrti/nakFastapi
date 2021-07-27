dev:
	docker build -t aws-base:latest ./
	docker run -d -p 80:80 -p 5000:5000 --name local-pause --cap-add=NET_ADMIN local-pause 
	docker-compose -f docker-compose.yml -f docker-compose-local.yml up --build


dev2:
	# 拉取镜像
	docker pull nikolaik/python-nodejs:python3.7-nodejs14-stretch
	# 启动并生成容器
	docker run -dit xxxx /bin/bash
	# 进入容器
	docker exec -it xxxx /bin/bash

copy:
	docker cp ./filder xxxx:opt

build:
    # 执行Doakerfile命令,生成新的镜像
	# docker build [ -t ｛イメージ名｝ [ :｛タグ名｝ ] ] ｛Dockerfileのあるディレクトリ｝
    docker build -t my-image:latest ./
	# 启动并生成容器
	docker run -dit xxxx /bin/sh
	# 进入容器
	docker exec -it xxxx /bin/sh

clean:
	# dockerで使用している容量の確認
	docker system df
	# Containerの全削除
	docker ps -aq | xargs docker rm
	# Imageの全削除
	docker images -aq | xargs docker rmi
	# Imageの全削除（強制）
	docker images -aq | xargs docker rmi -f

	# 清理 没有开启的  Docker 镜像
	docker system prune -a