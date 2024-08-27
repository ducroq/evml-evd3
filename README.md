# EVML-EVD3

This repository has the materials for the EVML-EVD3 course. The course is part of the minor Embedded Vision and Machine Learning.
The course teaches machine learning and deep learning for computer vision. It focuses on:

- How to apply these methods
- How to train models
- How to fine-tune models
- How to analyze performance

The course uses a hands-on approach. You start a machine learning project at the beginning of the semester.
To see the course topics and planning, refer to the [schedule](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/schedule.md).
You can add to our collection of applications. To do this, go to: https://padlet.com/jeroen_veen/zul8z8tbvhqpvb8t"

## Summary

EVML-EVD3 is a workshop that teaches machine learning and deep learning for (embedded) computer vision. These methods perform tasks without explicit instructions, helping systems adapt to changing conditions. They use data to estimate mathematical functions that model part of reality.

Many different machine learning algorithms exist, each with its own strengths. No single algorithm works best for all problems. Deep learning, a type of machine learning, tries to copy how the human brain works. It uses artificial neurons in multiple layers.

In this workshop, you will learn various machine learning approaches. You will study how to use neural networks, how to train and improve them, and how to analyze their performance. The course also covers data preparation and performance assessment.

The main goal is to teach you how to design, build, and test machine learning systems for image processing, with a focus on object classification. The course takes a hands-on approach. You will start a project at the beginning of the semester using Python packages like OpenCV, Scikit-learn, Tensorflow, and Keras.

For assessment, you will write a report about your project. In this report, you will explain your choices and evaluate your model's performance. This workshop provides practical skills for applying machine learning to real-world computer vision tasks."

## Learning objectives

You are expected to:
1. explain ML basic principles and reflect on its applications.
2. apply supervised ML in practice by selecting and training an algorithm, and preparing data.
3. analyze a supervised ML pipeline by validating and testing; and evaluating quality measures.
4. understand DL basic principles and reflect on its application.
5. apply a CNN in practice by selecting and training a network.
6. analyze a CNN and evaluates its performance.


## Materials

### Books
Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd ed.)*. O’Reilly Media (ISBN: 978-1492032649).

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.)*. O’Reilly Media (ISBN: 978-1098125974).

Kaehler, A. and Bradski, G. (2016). *Learning OpenCV 3*. O'Reilly Media (ISBN: 9781491937990).


### Resources
1. [schedule](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/schedule.md)
2. [sheets](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/sheets)
3. [scripts](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/scripts)
4. [support material](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/support%20material)
5. [templates](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/templates)
6. [collection of applications] (https://padlet.com/jeroen_veen/zul8z8tbvhqpvb8t)


## Assessment
During the course you will work on 2 mini projects. Furthermore, there will be theory quizzes. You will receive two grades, which are composed as:

1. Machine learning project report (80%) and quizzes (20%)
2. Deep learning project report (80%) and quizzes (20%)


### Mini projects
* A project team consists of 2-3 students
* Report building using templates (see [templates](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/templates))
* Deliver final results via HandIn as indicated in the [schedule](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/schedule.md)


### Quizzes
* Individual, multiple choice questions
* Online [socrative](http://www.socrative.com) room 1PTGB6PY
* Open book quiz, so books and slides can be consulted
* HAN student number, so NOT your name, nickname or anything else
* Quiz starts exactly at class hour and takes 10 minutes
* Be on time and have your equipment prepared
* During the quiz: no entering or leaving the classroom, and silence

## Schedule
The planning of the semester can be found in the [schedule](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/schedule.md). Here you can find the delivery deadlines and quiz occurences.


## Software development
You can download and install the following software:

* [A Git client](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html)
* [Python](https://www.python.org/downloads/)
* An IDE, such as [Visual Studio Code](https://code.visualstudio.com/), [Pycharm](https://www.jetbrains.com/pycharm/). [Data Spell](https://www.jetbrains.com/dataspell/).

### Python
Python is used extensively in this course. As a prerequisite you can test your skills using an online test, see e.g. [Python quiz](https://www.w3schools.com/quiztest/quiztest.asp?qtest=PYTHON). In addition, you can find many online tutorials that can help you to master Python, see e.g. [Python roadmap](https://roadmap.sh/python), [Learn Python](https://www.youtube.com/watch?v=rfscVS0vtbw), [Free Python books](https://github.com/EbookFoundation/free-programming-books/blob/main/books/free-programming-books-langs.md#python).


### Python packages
Either use the package manager of your IDE or use pip as a tool for installing Python packages, such as OpenCV, Scikit-learn, Tensorflow, and Keras.<br />
<code> pip install numpy scipy scikit-learn imutils opencv-python</code>


### Using GitLab
If you don't know how to use GitLab, you can simply download this repository as a ZIP archive. The downside is that you will have to check this repository for updates manually on a regular interval and merge changes by hand.<br />
If you would like to get started with GitLab, refer to the following [instructions](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html).<br />
<br />
As a reminder, here is a list of git command line commands that are often used:

* Clone via HTTPS (only the first time)<br />
  <code>git clone https://gitlab.com/jeroen_veen/evml-evd3.git</code>
* Receive changes from GitLab.com<br />
  <code>git pull origin master</code>


### Example scripts
During the lessons, multiple example [scripts](https://gitlab.com/jeroen_veen/evml-evd3/-/tree/main/scripts) will be discussed.


### Raspberry pi install
In the course, we will not run our models on a microcontroller, instead a Raspberry pi single board computer (SBC) is used. If you would like to get started with Raspberry pi, please folow these instructions:

* Raspberry Pi Imager (https://www.raspberrypi.org/downloads/)
* To enable SSH, create a file named ssh in boot partition
* Connect to [Eduroam](https://gitlab.com/jeroen_veen/evml-evd3/-/blob/main/support%20material/Raspberry%20Pi%20Debian%20Linux%20Wifi%20EDUROAM%20NL.pdf)
* To install opencv, see e.g. (https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/)
* In addition, numpy and scikit-learn are required. Find latest successfull builds for the current Raspbian distribution on [piwheels](https://www.piwheels.org/) and install using pip

