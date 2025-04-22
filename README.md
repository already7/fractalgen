git clone https://github.com/already7/fractalgen.git  
cd fractalgen  
mkdir pretrained_models  
Скачать веса https://drive.google.com/drive/folders/1fXErdyvgmjBNNWloIWJWzpWK730RDvfd?hl=ru и положить их в созданную папку  
docker build -f .devcontainer/Dockerfile -t myapp .  
docker run -it --rm --gpus all myapp  


#9.296109296109297  
#100  
