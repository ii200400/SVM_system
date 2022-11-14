// Include standard headers
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

// Include GLEWc
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
//cv
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// wand
//#include <MagickWand/MagickWand.h>
#include <Magick++.h>
using namespace Magick;

/*
VideoCapture cap1(0);
VideoCapture cap2(1);
Mat img_frame1;
Mat img_frame2;
*/

Mat cart_image;

float k1 = 0.2;
float k2 = 0.8;
float k3 = 20;

float rows = 765;
float cols = 765;

cv::Mat mapX(rows, cols, CV_32F);
cv::Mat mapY(rows, cols, CV_32F);

Mat getBowlImg();

void mapInit(cv::Mat mapX, cv::Mat mapY) {
	for (int x = 0; x < rows; x++) {
		for (int y = 0; y < cols; y++) {
			mapX.at<float>(x, y) = 2 * x / (cols - 1) - 1;
			mapY.at<float>(x, y) = 2 * y / (rows - 1) - 1;
		}
	}

	cv::Mat r(rows, cols, CV_32F);
	cv::Mat theta(rows, cols, CV_32F);

	cv::cartToPolar(mapX, mapY, r, theta);

	cv::Mat r2 = r.mul(r);
	cv::Mat r4 = r2.mul(r2);
	cv::Mat r6 = r2.mul(r4);

	cv::Mat temp = 1 + k1 * r2 + k2 * r4 + k3 * r6;
	r = r.mul(temp);

	cv::polarToCart(r, theta, mapX, mapY);
	mapX = ((mapX + 1) * cols - 1) / 2;
	mapY = ((mapY + 1) * cols - 1) / 2;
}

Mat wave(Mat image) {

	Mat result;

	cv::remap(image, result, mapX, mapY, cv::INTER_LINEAR);

	return result;

	Rect re(result.cols / 6, result.rows / 6, result.cols / 6 * 4, result.rows / 6 * 4);
	Mat returnImg = result(re).clone();

	return returnImg;
}

int main(void)
{
	InitializeMagick("C:/Users/multicampus/Desktop/ssafy/self_project/gitlab/SurroundViewMonitor");

	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1024, 768, "Tutorial 07 - Model Loading", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// Hide the mouse and enable unlimited mouvement
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set the mouse at the center of the screen
	glfwPollEvents();
	glfwSetCursorPos(window, 1024 / 2, 768 / 2);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);
	// 안쪽면만 그림
	glCullFace(GL_FRONT);

	cart_image = imread("./resource/image.jpg", IMREAD_COLOR);
	cvtColor(cart_image, cart_image, COLOR_BGR2RGB);

	GLuint VertexArrayID[2];
	glGenVertexArrays(2, VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders("TransformVertexShader.vertexshader", "TextureFragmentShader.fragmentshader");
	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	// Load the texture
	GLuint Texture = loadDDS("./resource/uvmap.DDS");
	// Get a handle for our "myTextureSampler" uniform
	GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");

	// 카트 유니폼
	GLuint Cart_MatrixID = glGetUniformLocation(programID, "Cart_MVP");
	GLuint Cart_Texture = loadDDS("./resource/sample.DDS");
	GLuint Cart_TextureID = glGetUniformLocation(programID, "Cart_myTextureSampler");


	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals; // Won't be used at the moment.
	bool res = loadOBJ("./resource/bowl.obj", vertices, uvs, normals);

	// 카트 오브젝트
	std::vector<glm::vec3> cart_vertices;
	std::vector<glm::vec2> cart_uvs;
	std::vector<glm::vec3> cart_normals;
	bool res2 = loadOBJ("./resource/cart.obj", cart_vertices, cart_uvs, cart_normals);

	// Load it into a VBO
	GLuint vertexbuffer[2];

	// 새로운 버퍼 생성 glGenBuffers(버퍼 개수, 이름 저장 공간)
	glGenBuffers(2, vertexbuffer);
	// 버퍼에 타겟을 할당(타겟, 버퍼)
	// GL_ARRAY_BUFFER -> 정점에 대한 데이터를 생성한 후 버퍼에 넣는다
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[0]);
	// 실제 버퍼에 데이터 넣기 (타켓, 사이즈, 실제넣을 데이터 주소값, usage)
	// GL_STATIC_DRAW -> 데이터가 저장되면 변경되지 않음
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[1]);
	glBufferData(GL_ARRAY_BUFFER, cart_vertices.size() * sizeof(glm::vec3), &cart_vertices[0], GL_STATIC_DRAW);


	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

	mapInit(mapX, mapY);

	bool img_init = false;
	Mat top;

	do {

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(programID);

		// Compute the MVP matrix from keyboard and mouse input
		computeMatricesFromInputs();
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 ModelMatrix = glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		// Set our "myTextureSampler" sampler to use Texture Unit 0
		glUniform1i(TextureID, 0);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		/*
		cap1 >> img_frame1;
		cap2 >> img_frame2;
		*/

		// 카메라에서 이미지 가져오는 함수 혹은 코드
		// VideoCapture cap1(1), cap1(2);

		// 왜곡을 없앤 이미지를 가져오는 함수
		// &frong, &right, &back, &left
		// getImg(&frong){};
		//Mat top = imread("./resource/test.png");
		//Mat top = getBowlImg(&frong, &right, &back, &left);

		if (!img_init) {
			top = getBowlImg();
			img_init = true;
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, top.cols, top.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, top.ptr());


		glBindVertexArray(VertexArrayID[0]);
		// 1rst attribute buffer : vertices
		// 위에 생성한 index 버퍼를 활성화
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[0]);
		// 저장한 데이터의 속성 정보 지정
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// Draw the triangle !
		glBindVertexArray(VertexArrayID[0]);
		glDrawArrays(GL_TRIANGLES, 0, vertices.size());


		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		glBindTexture(GL_TEXTURE_2D, Cart_Texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cart_image.cols, cart_image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, cart_image.data);

		glBindVertexArray(VertexArrayID[1]);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[1]);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);


		glBindVertexArray(VertexArrayID[1]);
		glDrawArrays(GL_TRIANGLES, 0, cart_vertices.size());

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);

	// Cleanup VBO and shader
	glDeleteBuffers(1, vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

// getBowlImg(&frong, &right, &back, &left)
Mat getBowlImg() {
	// 이미지 경로
	String absolute_path = "";
	String files[4] = { "front.png", "right.png", "left.png", "back.png"};

	Mat bowlImg;	// bowlView에 넣을 이미지
	Image img;	// 카메라에서 받아온 이미지
	double degree;	// 이미지를 회전시킬 정도 
	bool init = false;	// bowlImg 크기 초기화 여부

	// 파이썬의 "arc" 그냥 왜곡을 하는 방법 중 하나
	MagickCore::DistortMethod method = ArcDistortion;
	
	for (int i = 0; i < 4; i++) {
		// 이미지 불러오기
		img = Image(absolute_path + files[i]);

		if (i == 0){
			degree = 315;
		}else if (i == 1) {
			degree = 45 + 4;
		}else if (i == 2) {
			degree = 225 - 4;
		}else{
			degree = 135;
		}

		// 아래의 파라미터에 따라서 왜곡 진행
		double listOfArguments[2] = { 90, degree };
		img.distort(method, 2, listOfArguments);

		// 이미지 크기
		const size_t  w = img.columns(), h = img.rows();

		if (!init) {
			// zero함수를 사용하면 Scalar(0)으로 고정되어 3차원 배열이 불가능하다.
			bowlImg = Mat(w * 1.5, h * 1.5, CV_8UC3, Scalar(0, 0, 0));
			init = true;
		}

		// Image 클래스의 각 픽셀 정보를 가져오기 위한 변수
		Quantum *pixel_cache = img.getPixels(0, 0, w, h);

		// bowlView 변수에 왜곡을 넣은 이미지 붙여넣기

		int x_fin = h;
		int y_fin = w;
		int x0 = x_fin - 350;
		int y0 = y_fin - 350;

		if (i == 0) {
			Mat temp = Mat(h, w, CV_8UC3, Scalar(255, 255, 255));
			img.write(0, 0, w, h, "BGR", Magick::CharPixel, temp.data);

			Mat imageROI = bowlImg(Rect(0, 0, y0, x0));
			temp = temp(Rect(0, 0, y0, x0));

			addWeighted(imageROI, 1.0, temp, 1.0, 0.0, imageROI);
		}
		else if (i == 1) {
			int x1 = 10;
			int y1 = (int)(y_fin / 4) - 25;

			int x1_move = 0;
			int y1_move = 25;

			int x1_size = x_fin - x1;
			int y1_size = y_fin - y1;

			Mat temp = Mat(h, w, CV_8UC3, Scalar(255, 255, 255));
			img.write(0, 0, w, h, "BGR", Magick::CharPixel, temp.data);

			Mat imageROI = bowlImg(Rect(y0 + y1_move, x1_move, y1_size, x1_size));
			temp = temp(Rect(y1, x1, y1_size, x1_size));

			addWeighted(imageROI, 1.0, temp, 1.0, 0.0, imageROI);
			imshow("test", bowlImg);
		}
		else if (i == 2) {
			int x2 = (int)(x_fin / 4 - 40);
			int y2 = (int)(0);

			int x2_move = 0;
			int y2_move = 30;

			y_fin = int(y_fin * 3 / 4);

			int x2_size = x_fin - x2;
			int y2_size = y_fin - y2;

			Mat temp = Mat(h, w, CV_8UC3, Scalar(255, 255, 255));
			img.write(0, 0, w, h, "BGR", Magick::CharPixel, temp.data);		

			Mat imageROI = bowlImg(Rect(y2_move, x0 + x2_move, y2_size, x2_size));
			temp = temp(Rect(y2, x2, y2_size, x2_size)).clone();

			addWeighted(imageROI, 1.0, temp, 1.0, 0.0, imageROI);
		}
		else {
			int x3 = (int)(x_fin / 4);
			int y3 = (int)(y_fin / 4 - 30);

			int x3_move = 0;
			int y3_move = 0;

			int x3_size = x_fin - x3;
			int y3_size = y_fin - y3;

			Mat temp = Mat(h, w, CV_8UC3, Scalar(255, 255, 255));
			img.write(0, 0, w, h, "BGR", Magick::CharPixel, temp.data);

			Mat imageROI = bowlImg(Rect(y0 + y3_move, x0 + x3_move, y3_size, x3_size));
			temp = temp(Rect(y3, x3, y3_size, x3_size)).clone();

			addWeighted(imageROI, 1.0, temp, 1.0, 0.0, imageROI);
		}

	}

	imshow("top", bowlImg);
	return bowlImg;
}