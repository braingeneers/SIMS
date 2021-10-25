CONTAINER = jmlehrer/cell-exploration

exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .

push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

go:
	make build && make push

train:
	python src/models/run_model_search.py
