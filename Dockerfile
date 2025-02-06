FROM dolfinx/dolfinx:stable

RUN apt-get update && \
      apt-get -y install sudo

RUN apt-get install --fix-missing -y xvfb

RUN pip install matplotlib pyvista[jupyter] trame trame-vtk jupyter imageio

# Set the working directory to /app
WORKDIR /app

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
