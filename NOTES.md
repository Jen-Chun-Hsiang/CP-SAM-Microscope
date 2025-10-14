# BUILD DOCKER LOCALLY 
docker logout   
docker login -u emilyhsiang
(go to passkey to find the password)
export DH_IMG=docker.io/emilyhsiang/cellpose:cu12-2025-10-10
docker buildx build --platform linux/amd64 --build-arg GPU=1 -t $DH_IMG --push .