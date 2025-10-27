#!/bin/bash



cd cicflowmeter-docker-master

docker build -t cicflowmeter .

cd ..

docker-compose build
docker-compose up -d

docker exec dos_attacker /tools/dos_attack.sh

docker-compose stop tcpdump

docker run --rm --log-driver=none \
  -v $(pwd)/pcaps:/input \
  -v $(pwd)/flows:/output \
  cicflowmeter \
  /input/dos_attack.pcap /output >/dev/null 2>&1

rm pcaps/dos_attack.pcap 

python ids.py

docker-compose down