# Instrumented Objects for Assessing Compliant Robotic Grasping
Project page for the paper "Towards assessing compliant robotic grasping from first-object perspective via instrumented objects", which is under review for RA-L.
```
@article{knopke2023towards,
  title={Towards Assessing Compliant Robotic Grasping from First-Object Perspective via Instrumented Objects},
  author={Knopke, Maceon and Zhu, Liguo and Corke, Peter and Zhang, Fangyi},
  journal={arXiv preprint arXiv:2312.14466},
  year={2023}
}
```

The design and implementations will be open-sourced soon. Please refer to the paper (https://arxiv.org/abs/2312.14466) for more details.

[![Watch the video](https://img.youtube.com/vi/kQSZlNxYRrs/0.jpg)](https://www.youtube.com/watch?v=kQSZlNxYRrs)

CAD_Models:
-The file Internal_3_hall_mould contains the two part mould required to create the silicone shell of the sensor. These two parts clip together and mixed 2-part/platinum cure silicone can be poured in. It is recommended to place the filled mould in a vacuum chamber to remove bubbles from within the silicone during the curing process. 

-The file Internal_multi_sensor_core is the core which is inserted into the shell. The supplied pcb design fits into the surface of the core and attached wires are routed into the center of each face and through the core.

-The files Probe_x contain the probes used during data collection for the paper. These were mounted onto a vernier dual force sensor during initial calibration and onto a Franka Emika Gripper for Panda for data collection.

-The file TMAG5273A1_PCB_mini is a cad model of the PCB and attached TMAG5273A1 hall effect sensor and capacitor. 

PCB:
- The supplied pcb can be opened using kicad for editing. This pcb holds a single TMAG5273A1 sensor and capacitor. Wires are inserted into the through holes and soldered. Reommend gluing the fibers to the pcb for strain relief. The folder GBR_Files contains the files required for printing via PCBway.

Images:
-Various images from the paper and some extas are supplied to demonstrate the assembly and testing setups used.

Arduino Code:
The code provided was used in multiple experiments, it is set up to program and read data from a 3 sensor prototype, and a 15 sensor prototype. It runs on an arduino mega and can be connected to a cnc machine that accepts grbl to control the probes if required. Data from the sensors is printed over serial and putty was used to save the large amounts of incoming data.


*Notes:
The paper details the assembly process, if further information is required please contact fangyi.zhang@qut.edu.au and a more detailed guideline can be created.

The current design can be improved upon by switching to a sensor that remembers programmed i2c addresses. The TMAG5273A1 only has 4 factory addresses and new addresses must be assigned on startup to use multiple simultaneously. This required an additional power wire to power on each sensor individually during startup for programming. Using a sensor that recalls its programmed address on boot will simplify the wiring significantly. Other methods could be the use of multiplexers or alternative communication protocols. If the number of required GPIO is reduced an internal microcontroller with wireless capability can be used such as an esp32s3.
