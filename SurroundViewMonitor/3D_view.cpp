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
		Mat top = getBowlImg();

		imshow("top", top);

		//Mat result = wave(top);
		//imshow("circle", result);


		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, result.cols, result.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, result.ptr());
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
	String files[4] = { "front.png", "right.png", "back.png", "left.png"};

	Mat bowlImg;	// bowlView에 넣을 이미지
	Image img = Image("back.png");	// 카메라에서 받아온 이미지
	double degree;	// 이미지를 회전시킬 정도 
	bool init = false;	// bowlImg 크기 초기화 여부

	// 파이썬의 "arc" 그냥 왜곡을 하는 방법 중 하나
	MagickCore::DistortMethod method = ArcDistortion;
	
	for (int i = 0; i < 4; i++) {
		// 이미지 불러오기
		//imread(absolute_path + files[i]);
		// (image=None, blob=None, file=None, filename=None, pseudo=None, background=None, colorspace=None, depth=None, extract=None, format=None, height=None, interlace=None, resolution=None, sampling_factors=None, units=None, width=None)
		//int img = Magick::Image(NULL, absolute_path + files[i]);
		img = Image(absolute_path + files[i]);
		
		//printf("width : %d, height: %d\n", w, h);

		if (i == 0){
			degree = 315;
		}else if (i == 1) {
			degree = 45;
		}else if (i == 2) {
			degree = 135;
		}else{
			degree = 225;
		}

		// 아래의 파라미터에 따라서 왜곡 진행
		double listOfArguments[2] = { 75, degree };
		img.distort(method, 2, listOfArguments);

		// 이미지 크기
		const size_t  w = img.columns(), h = img.rows();
		if (!init) {
			// https://3001ssw.tistory.com/172
			// zero함수를 사용하면 Scalar(0)으로 고정되어 3차원 배열이 불가능하다.
			bowlImg = Mat(w * 2, h * 2, CV_8UC3, Scalar(0, 0, 0));
			init = true;
		}

		int sub = 310;
		int x = w - sub;
		int y = h - sub;
		int x2 = w;
		int y2 = h;

		// https://www.imagemagick.org/Magick++/Image++.html#Raw%20Image%20Pixel%20Access
		// Low-Level Image Pixel Access 참고
		// https://stackoverflow.com/questions/47781396/imagemagick-c-version-7-modify-pixel-value-in-blank-image
		//img.modifyImage(); // 필요 없는 듯..?

		// Image 클래스의 각 픽셀 정보를 가져오기 위한 변수
		Quantum *pixel_cache = img.getPixels(0, 0, w, h);

		// bowlView 변수에 왜곡을 넣은 이미지 붙여넣기 (수동으로..)
		int bowl_h, bowl_w, img_h, img_w;
		if (i == 0) {
			bowl_h = 0, bowl_w = 0;
			img_h = 0, img_w = 0;
		}
		else if (i == 1) {
			bowl_h = 0, bowl_w = y;
			img_h = 0, img_w = sub;
		}
		else if (i == 2) {
			bowl_h = x, bowl_w = y;
			img_h = sub, img_w = sub;
		}
		else {
			bowl_h = x, bowl_w = 0;
			img_h = sub, img_w = 0;
		}

		// RGB이니 당연히 3이다.
		int channels = img.channels();
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				std::size_t offset = (w * (i + img_h) + (j + img_w));
				std::size_t moffset = channels * offset;
				// 나누는 이유
				// https://stackoverflow.com/questions/47781396/imagemagick-c-version-7-modify-pixel-value-in-blank-image
				// 놀랍게도 RGB가 아니라 BGR인 것 같..다. 아래 글을 보니 맞다.
				// https://stackoverflow.com/questions/7899108/opencv-get-pixel-channel-value-from-mat-image
				Vec3b color = Vec3b(
					pixel_cache[moffset + 2] / 257,
					pixel_cache[moffset + 1] / 257,
					pixel_cache[moffset + 0] / 257);

				/*if (pixel_cache[moffset + 0] > 0) {
					cout << pixel_cache[moffset + 0] << " "<< pixel_cache[moffset + 0] / 257 << endl;
					cout << pixel_cache[moffset + 1] << " " << pixel_cache[moffset + 1] / 257 << endl;
					cout << pixel_cache[moffset + 2] << " " << pixel_cache[moffset + 2] / 257 << endl;
				}*/
				bowlImg.at<Vec3b>((i + bowl_h), (j + bowl_w)) = color;
				//Vec3b colortemp = bowlImg.at<Vec3b>(x, y);
			}
		}

		
		/*
        if i==0:
            result[0:x,0:y]=np.asarray(img)[0:x,0:y]
        if i==1:
            result[0:x,y:y*2]=np.asarray(img)[0:x,sub:y2]
        if i==2:
            result[x:x*2,y:y*2]=np.asarray(img)[sub:x2,sub:y2]
        if i==3:
            result[x:x*2,0:y]=np.asarray(img)[sub:x2,0:y]
		*/
	}

	imshow("top", bowlImg);
	//imwrite(absolute_path + "result5.png", result[0:result.shape[0] - sub * 2, 0 : result.shape[1] - sub * 2]);

	return bowlImg;
}