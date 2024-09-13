FROM continuumio/miniconda3

# This prevents Python from writing out pyc files
ENV PYTHONUNBUFFERED 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install curl -y && apt-get install -y gcc
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && apt-get install -y nodejs

WORKDIR /app/backend

COPY requirements.yml /app/backend/

# Build psycopg2-binary from source -- add required required dependencies
RUN . ~/.bashrc && conda update conda
RUN /opt/conda/bin/conda env create -f /app/backend/requirements.yml
ENV PATH /opt/conda/envs/C4Qenv/bin:$PATH

RUN echo "source activate C4Qenv" >~/.bashrc

#COPY ./scripts /scripts
#RUN chmod +x /scripts*

COPY . /app/backend/


CMD [ "python", "manage.py", "makemigrations"]
CMD [ "python", "manage.py", "migrate"]
CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]