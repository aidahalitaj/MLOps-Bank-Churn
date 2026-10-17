FROM python:3.10
ENV PYTHONPATH=/app
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -U pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
RUN pip install -r requirements-deploy.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
COPY . .
EXPOSE 8080
CMD uvicorn src.app.main:app --host 0.0.0.0 --port 8080