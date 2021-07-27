FROM python:3.8-alpine

ENV GLIBC_VER=2.31-r0


# dockerのインストール 
# pipのインストール
# nodeのインストール
# makeのインストール
# docker-composeのインストール
# gitのインストール
RUN apk add docker \
    && apk add --update py-pip \
    && apk add --update nodejs npm \
    && apk add --update make \
    && apk add docker-compose \
    && apk add --no-cache git