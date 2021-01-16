FROM pytorch/pytorch:latest

WORKDIR /workspace

COPY vergeai-python /workspace/vergeai-python
COPY pytorch-serde /workspace/pytorch-serde
COPY files/ /workspace/

RUN pip install ./vergeai-python -U
RUN pip install ./pytorch-serde -U

CMD ["python3", "server.py"]
