CONTAINER = jmlehrer/sims

.PHONY: build exec push run go train release 
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

release:
	rm -rf dist/ && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/* --verbose
