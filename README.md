# Deep-Learning-Project
This is a research project for helping dentists automatically mark the facial landmarks which will be used in the treatment or surgery with deep learning and python programming.

These are the landmarks that we want to find:</br>
<img width="564" alt="image" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/a4470433-fa97-47fb-9eaf-528cbf077f73"></br>
The overall structure of my research:</br>
<img width="811" alt="image" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/7d91ea4d-7a7f-46cb-bbfc-59d4fa5ca807"></br>
The first stage:</br>
We cropped the original picture into five different areas to help training to find the face area with a more accurate detection rate:</br>
<img width="929" alt="Screenshot 2023-07-15 at 3 36 52 PM" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/135d0f7f-d159-44b9-87e0-4f9715b79376"></br>
This is the framework of the first stage:</br>
<img width="862" alt="image" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/c8eb1c24-2bb7-45ee-a61d-2d3f69d68abc">
The second stage:</br>
After getting the detected face area, we can try to find the Landmark 10 via the second stage with Euclidean distance loss function:</br>
<img width="868" alt="image" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/38a15c8e-343d-4418-8c19-2e3fdc434511">
Before the last stage:</br>
We implemented data augmentation to get more training data to make our detection more precise. We generated 30 more pictures for each by moving the original picture in the range of [-50,50].</br>
The examples of data augmentation:</br>
<img width="379" alt="Screenshot 2023-07-15 at 3 46 58 PM" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/2c008184-e30c-4a9e-9bc7-10a0860ece23"></br>
The last stage:</br>
We detected the final L10 landmark with the last stage.</br>
<img width="885" alt="image" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/aa0715cf-3803-4ce4-b591-198799bcb5ac"></br>
</br>
The experimental results:</br>
<img width="625" alt="image" src="https://github.com/s890126/Deep-Learning-Project/assets/65753398/d665578c-adaf-4887-8826-d9079dcf25a4"></br>




