project_name := femin

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(patsubst %/,%,$(dir $(mkfile_path)))

build:
	docker build -t $(project_name) .

run:
	docker run -it -p 8888:8888 -v $(current_dir)/femin:/app --rm --name $(project_name) $(project_name)
