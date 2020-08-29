torch-model-archiver --model-name torch_model --version 1.0 --model-file servefiles_finals/model.py --serialized-file servefiles_finals/weights.pt --handler servefiles_finals/handler.py
mv torch_model.mar serve/model_store/
cd serve
torchserve --start --ts-config config.properties
