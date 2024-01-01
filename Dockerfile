FROM python:3.10
COPY . /app
WORKDIR /app/book_impact
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN ls /app/
# RUN mkdir /app/data/processed/
ENV PYTHONPATH=/app
# RUN wget https://drive.google.com/file/d/1VXp7KDhIvh2B7nnsBdfeMY4SwMNrPtCs/view?usp=sharing
CMD ["python", "scripts/train_text_encoder.py", "--phase", "train", "-m", "local[2]", "-i", "data/raw/books_task.csv", "-t", "data/processed/train/text_features"]
