FROM python
ADD . /src/.
WORKDIR /src
RUN pip3 install -r requirements.txt
