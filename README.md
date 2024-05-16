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

Please refer to the paper (https://arxiv.org/abs/2312.14466) for more details.

[![Watch the video](https://img.youtube.com/vi/kQSZlNxYRrs/0.jpg)](https://www.youtube.com/watch?v=kQSZlNxYRrs)

For fabrication details, please refer to the [fabrication readme](Fabrication_Readme.md).

For the contact estimation, codes and data are in [Contact_Estimation](Contact_Estimation).

The full results for the other four faces can be seen in [detailed_results_for_the_other_four_faces.pdf](detailed_results_for_the_other_four_faces.pdf).


## Other Machine Learning Methods for Contact Estimation
![Neural Networks vs Linear Regression](method_comp.png)

A comparison was made to a baseline linear regression method, whose results are shown in the above table. The linear regression has very poor performance in all metrics, indicating it is not suitable for modeling such complex non-linear mapping from Hall-effect signals to contact forces. The fully-connected network used in this work is just a proof of concept for data-driven contact estimation via Hall-effect signals. We do not claim the optimality of the neural network used in the paper and it is worth exploring more appropriate machine learning methods for different observations in future works and all feasible solutions will help scale the application of instrumented compliant objects for robotic grasping performance assessment.

## Training Data Collection: Balancing the spatial and temporal coverage of training data
As discussed in Section VII, studies in Section V-E and Section V-F showed that the location coverage of training data is critical for better contact location/case estimation accuracy, while much less data (2683 samples in total, and about 19 samples for each contact location/case) can enable accurate contact force estimation. In future training/calibration data collection, a better balance can be achieved by collecting less data for each contact location/case but covering as many contact locations and numbers of contacts as possible.

## Other Designs
As introduced in the Introduction of the [paper](https://arxiv.org/abs/2312.14466), this work is motivated to assess robotic grasping of compliant objects by measuring the stress or damage from a first-object perspective using mock objects (instrumented compliant objects). From a mock-compliant object perspective, the possibility of extending a design for different properties (such as hardness and shapes) and for providing different observations (such as contact force, deformation, and shape changes) is critical. Therefore, as a proof of concept, this paper adopts:
1) the Hall-sensor-based silicone design which can be adapted to different hardness and shapes and for observing 2D or 3D contact forces, deformation, and shape changes; and 
2) the machine learning approach which can model various input-output mappings if with required training data and unique input-output pairs.

However, as discussed in the [paper](https://arxiv.org/abs/2312.14466), we do not claim the current design’s optimality. More designs are worth exploring in future works and all feasible designs (including variants of the work by Kıiva et al) will help to extend the application of instrumented objects for assessing the grasping of various compliant objects. If only 2D tactile pressure is needed, some off-the-shelf sensor films could be a good option, such as a tactile pressure indicating sensor film (https://www.sensorprod.com/index.php).

## Control of the Object's Hardness
As discussed in Section IV in the [paper](https://arxiv.org/abs/2312.14466), if more or less deformation is needed, different properties including the hardness of the instrumented object can be achieved by using different materials such as silicone with different levels of hardness for the outer shell. By adjusting this hardness, the deformation measured by the sensors can be tailored to match specific scenarios. The Shore hardness scale ranges from soft rubbers and gels (Shore 00), soft to semi-rigid rubbers (Shore A) and hard rubbers/plastics (Shore D). In this paper, all experiments utilized a shell made from Platsil GEL-10 Prosthetic Grade Silicone, which has a Shore hardness of A10. Shore A10 was chosen because it provides firmness, simplifying prototype design and testing, while still allowing elastic deformation under reasonable force (e.g., pinching or firm pressing). This hardness can produce deformation for detecting as little as 0.12N contact force changes. Switching to a silicone with lower Shore hardness (e.g., Shore 0010-0030) is expected to increase sensitivity, but would result in a lower readable maximum force. Increasing the hardness would decrease the sensitivity, while allowing for a higher range of applied force. Conversely, increasing hardness too much into the mid Shore D range could lead to inaccurate readings due to deformation of the plastic inner core and PCBs. 

While silicone is suggested due to its elastic properties and ease of use, alternative elastomers could also be used, such as Thermoplastic Polyurethane (TPU), which can be 3D printed using FDM and SLS technologies. TPU offers a large hardness range, commonly measured at Shore 60A-90A, although some variants can be as low as Shore 0030. Elastic resins are also available for resin 3D printers, with a hardness of around Shore 50-80A.

## Limitations and Future Outlook
The current design generally worked well in contact estimation, although there were some design drawbacks that caused non-trivial performance degradation with simultaneous contacts from multiple faces and the internal core shifting w.r.t. the silicone shell between calibration and evaluation.

This concept and design can be extended in the future for more comprehensive measurements of compliant robotic grasping (such as the measurement of 3D contact force and more simultaneous contact points, and the estimation of deformation and shape changes) with various robotic grippers (such as anthropomorphic robotic hands and soft robotic grippers/hands). Ultimately this could lead to a robust framework for autonomously generating designs of instrumented objects with various measurement capabilities and physical properties (shapes, sizes, stiffness, etc) for benchmark purposes. Such instrumented objects could close the loop of developing/learning solutions for different compliant robotic manipulation tasks by providing timely and accurate feedback from a first-object perspective.

### Use for assessing soft robotic grippers/hands
To be used for assessing soft robotic grippers/hands, the design might need to be adjusted in three aspects:
- collecting additional training data to cover more complicated contact situations with the object,
- denser sensors or more effective sensor layouts for detecting more complicated contacts (e.g., large contact areas that cover many more than three pixels), and
- a softer outer shell to be more sensitive to subtle deformations (depending on how compliant the grippers/hands are).

### Use for assessing grasping by anthropomorphic robotic hands
This instrumented object can be extended for assessing grasping by anthropomorphic robotic hands, but needs additional training data to cover situations different from the three types of probes currently used for training data collection. For better and stabler performance, the current design also needs to be optimized to eliminate the Problems a) and b) discussed in Section VII in the [paper](https://arxiv.org/abs/2312.14466).