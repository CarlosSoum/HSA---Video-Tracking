// Para compilar:
// g++ fitness.c -lOpenCL -o fitness `pkg-config --cflags --libs opencv`

// OpenCV (Visión Computacionañ)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// OpenCL (Computo Paralelo)
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// Otras
#include <math.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>

// Configuración de GPU
std::string kernel_file = "fitness_kernel.c";
std::string kernel_file_bee = "fitness_kernel_bee.c";
std::string iou_route = "results/IOU_Results.txt";
char *kernel_src;
long kernel_size;
int err;
cl_device_id device_id;
cl_context context;
cl_command_queue commands;
cl_program program;
cl_kernel kernel;
size_t max_work_group_size;
size_t localWorkSize[2], globalWorkSize[2];

// Configuración de dataset ALOV300++
std::string path_alov;
std::string img_dir;
std::string ann_dir;
std::string vid_name;
std::string check_str;
int vid_num;
double max_window;
short u, v, window, initial_frame, last_frame, u0, v0, w0, h0, u2, v2, n2, m2;
double img_ratio = 1.0;
std::string path_result;
std::ofstream result_file;
int use_sobel = 0;
int resting = 0;
int honeybee = 1;
int resizea = 0;
int save_track = 0;
int take_metrics = 0;
int gammaSize;

// Configuración del algoritmo HSA
int cores_per_bee;
int num_bees;
short max_gen;
int eta_m;
int eta_c;
float rate_alpha_e;
float rate_beta_e;
float rate_gamma_e;
float rate_alpha_r;
float rate_beta_r;
float rate_gamma_r;
float _rate_alpha_e;
float _rate_beta_e;
float _rate_gamma_e;
float _rate_alpha_r;
float _rate_beta_r;
float _rate_gamma_r;
CvRNG rng = cvRNG(0xffffffff);

// Obtención de valor beta para cruce Cruce Binario Simulado (SBX)
double get_beta(double u, double eta_c) {
	double beta;
	u = 1 - u;
	double p = 1.0 / (eta_c + 1.0);

	if (u <= 0.5) {
		beta = pow(2.0 * u, p);
	}
	else {
		beta = pow((1.0 / (2.0 * (1.0 - u))), p);
	}
	return beta;
} 

// Obtención de valor delta para mutación polinomial
double get_delta(double u, double eta_m) {
	double delta;
	if (u <= 0.5) {
		delta = pow(2.0 * u, (1.0 / (eta_m + 1.0))) - 1.0;
	} 
	else {
		delta = 1.0 - pow(2.0 * (1.0 - u), (1.0 / (eta_m + 1.0)));
	}
	return delta;
}

// Lectura del archivo de configuración (config)
void read_config() {
	std::string trash;
	std::ifstream in("./config");
	in >> trash;
	in >> path_alov;
	in >> trash;
	in >> path_result;
	in >> trash;
	in >> img_dir;
	in >> trash;
	in >> ann_dir;
	in >> trash;
	in >> vid_name;
	in >> trash;
	in >> vid_num;
	in >> trash;
	in >> max_window;
	in >> trash;
	in >> cores_per_bee;
	in >> trash;
	in >> max_gen;
	in >> trash;
	in >> eta_m;
	in >> trash;
	in >> eta_c;
	in >> trash;
	in >> _rate_alpha_e;
	in >> trash;
	in >> _rate_beta_e;
	in >> trash;
	in >> _rate_gamma_e;
	in >> trash;
	in >> _rate_alpha_r;
	in >> trash;
	in >> _rate_beta_r;
	in >> trash;
	in >> _rate_gamma_r;
	in >> trash;
	in >> path_result;
	in.close();

	std::cout << "\n" << "Reading config file: " << "\n";
	std::cout << "Path ALOV++: " << path_alov << "\n";
	std::cout << "Path result: " << path_result << "\n";
	std::cout << "Image dir: " << img_dir << "\n";
	std::cout << "Annotation dir: " << ann_dir << "\n";
	std::cout << "Video name: " << vid_name << "\n";
	std::cout << "Video number: " << vid_num << "\n";
	std::cout << "Max window size: " << max_window << "\n";
	if (honeybee == 1) {
		std::cout << "Cores per bee: " << cores_per_bee << "\n";
		std::cout << "Max generation: " << max_gen << "\n";
		std::cout << "Eta for mutation: " << eta_m << "\n";
		std::cout << "Eta for crossover: " << eta_c << "\n";
		std::cout << "Alpha for exploration: " << rate_alpha_e << "\n";
		std::cout << "Beta for exploration: " << rate_beta_e << "\n";
		std::cout << "Gamma for exploration: " << rate_gamma_e << "\n";
		std::cout << "Alpha for foraging: " << rate_alpha_r << "\n";
		std::cout << "Beta for foraging: " << rate_beta_r << "\n";
		std::cout << "Gamma for foraging: " << rate_gamma_r << "\n";

		std::cout << "Max window / sqrt(cores per bee): " << 
		(max_window / sqrt(cores_per_bee)) << " (Should be integer)\n\n";
	}
}

// Lectura de kernel OpenCL
long read_kernel(std::string path, char **buf) {
	FILE *fp;
	size_t fsz;
	long off_end;
	int rc;

	// Apertura del archivo
	fp = fopen(path.c_str(), "r");
	if(NULL == fp) {
		return -1L;
	}

	// Búsqueda del final del archivo
	rc = fseek(fp, 0L, SEEK_END);
	if(0 != rc) {
		return -1L;
	}

	// Desplazamiento de bytes hasta el final del archivo (tamaño)
	if(0 > (off_end = ftell(fp))) {
		return -1L;
	}
	fsz = (size_t)off_end;

	// Asignación archivo en búfer provisional
	*buf = (char *) malloc(fsz + 1);
	if(NULL == *buf) {
		return -1L;
	}

	// Recolocación del puntero hasta el inicio del archivo
	rewind(fp);

	// Extracción del archivo desde el buffer provisional
	if( fsz != fread(*buf, 1, fsz, fp) ) {
		free(*buf);
		return -1L;
	}

	// Cerrar el archivo
	if(EOF == fclose(fp)) {
		free(*buf);
		return -1L;
	}

	// Asegurarse de que el buffer provisional se encuentra vacío, solo por si acaso
	(*buf)[fsz] = '\0';

	// Regresar el tamaño del archivo
	return (long)fsz;
}

// Preparación de la GPU
void set_gpu() {
	printf("GPU preparation:\n");

	// Conexión al dispositivo GPU
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);
	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0],
		gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group!\n");
		exit(1);
	}

	// Revisar límites del dispositivo
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(max_work_group_size), &max_work_group_size, NULL);
	printf("Max work group size: %lu\n", max_work_group_size);

	// Obtener tamaño de la subpoblación
	num_bees = max_work_group_size / cores_per_bee;
	printf("Number of bees: %d\n", num_bees);
	rate_alpha_e = num_bees * _rate_alpha_e;
	rate_beta_e = num_bees * _rate_beta_e;
	rate_gamma_e = num_bees * _rate_gamma_e;
	rate_alpha_e = round(rate_alpha_e);
	rate_beta_e = round(rate_beta_e);
	rate_gamma_e = round(rate_gamma_e);

	rate_alpha_r = num_bees * _rate_alpha_r;
	rate_beta_r = num_bees * _rate_beta_r;
	rate_gamma_r = num_bees * _rate_gamma_r;
	rate_alpha_r = round(rate_alpha_r);
	rate_beta_r = round(rate_beta_r);
	rate_gamma_r = round(rate_gamma_r);

	// Resta para obtención de la población alfa
	if (rate_alpha_e + rate_beta_e + rate_gamma_e != num_bees) {
		rate_alpha_e += num_bees - 
			(rate_alpha_e + rate_beta_e + rate_gamma_e);
	}
	if (rate_alpha_r + rate_beta_r + rate_gamma_r != num_bees) {
		rate_alpha_r += num_bees - 
			(rate_alpha_r + rate_beta_r + rate_gamma_r);
	}

	// Parificación del valor beta
	if ((int)rate_beta_e % 2 != 0) {
		rate_alpha_e++;
		rate_beta_e--;
	}
	if ((int)rate_beta_r % 2 != 0) {
		rate_alpha_r++;
		rate_beta_r--;
	}
	if (honeybee == 1) {
		printf("Alpha e: %f\n", rate_alpha_e);
		printf("Beta e: %f\n", rate_beta_e);
		printf("Gamma e: %f\n", rate_gamma_e);
		printf("Alpha r: %f\n", rate_alpha_r);
		printf("Beta r: %f\n", rate_beta_r);
		printf("Gamma r: %f\n\n", rate_gamma_r);
	}

	// Creación de contexto computacional para computo paralelo
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		printf("Error: Failed to create a compute context!\n");
		exit(1);
	}

	// Creación de comando de comandos para función kernel paralela
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) {
		printf("Error: Failed to create a command commands!\n");
		exit(1);
	}

	// Creación de programa computable a partir del archivo fuente del kernel
	if(kernel_size < 0L) {
		perror("Kernel file read failed");
		exit(1);
	}
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src, NULL, &err);
	if (!program) {
		printf("Error: Failed to create compute program!\n");
		exit(1);
	}

	// Compilación del programa ejecutable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, 
			CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Creación de kernel computable en el programa que deseamos ejecutar
	kernel = clCreateKernel(program, "fitness", &err);
	if (!kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}
}

// Función que procesa información del archivo de anotación y obtiene información para la obtención del IoU (Intersection Over Union)
void extractIoUInfo(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy) {
	u2 = round(std::min(ax, std::min(bx, std::min(cx, dx))));
	v2 = round(std::min(ay, std::min(by, std::min(cy, dy))));
	n2 = round(std::max(ax, std::max(bx, std::max(cx, dx))) - 
		std::min(ax, std::min(bx, std::min(cx, dx))));
	m2 = round(std::max(ay, std::max(by, std::max(cy, dy))) - 
		std::min(ay, std::min(by, std::min(cy, dy))));
}

// Lectura del archivo de anotación
void datsread_ann() {
	double ax, ay, bx, by, cx, cy, dx, dy;

	// Abrir el archivo
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	const std::string& ann_path = path_alov + ann_dir + vid_name + '/' +
		vid_name + "_video" + vid_num_str + ".ann";
	FILE* ann_file = fopen(ann_path.c_str(), "r");

	// Leer el archivo
	fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&initial_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);
	u0 = round(std::min(ax, std::min(bx, std::min(cx, dx))));
	v0 = round(std::min(ay, std::min(by, std::min(cy, dy))));
	w0 = round(std::max(ax, std::max(bx, std::max(cx, dx))) - 
		std::min(ax, std::min(bx, std::min(cx, dx))));
	h0 = round(std::max(ay, std::max(by, std::max(cy, dy))) - 
		std::min(ay, std::min(by, std::min(cy, dy))));
	window = std::max(w0, h0);

	while (fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&last_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy) != EOF) {}

	// Cerrar el archivo
	fclose(ann_file);

	std::cout << "Reading annotation file: " << "\n";
	std::cout << "u0: " << u0 << "\n";
	std::cout << "v0: " << v0 << "\n";
	std::cout << "Real window: " << window << "\n";
	std::cout << "First frame: " << initial_frame << "\n";
	std::cout << "Last frame: " << last_frame << "\n\n";
}

// Lectura de un frame del video
cv::Mat read_frame(int frame_num) {
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	char frame_num_str[8];
	sprintf(frame_num_str, "%08d", frame_num);
	const std::string& vid_path = path_alov + img_dir + vid_name + '/' +
		vid_name + "_video" + vid_num_str + "/" + frame_num_str + ".jpg";
	return cv::imread(vid_path, CV_LOAD_IMAGE_COLOR);
}

// Guardado de una imagen resultante a partir del análisis de un frame
void save_img(cv::Mat m, int frame_num_m, std::string info, int target) {
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	char frame_num_str[8];
	sprintf(frame_num_str, "%08d", frame_num_m);

	cv::Point p1 = cvPoint(u0, v0), p2 = cvPoint(u0 + w0, v0 + h0);
	cv::Scalar red = CV_RGB(255, 0, 0); 
	cv::rectangle(m, p1, p2, red);
	if (target) {
		cv::Point p3 = cvPoint(u2, v2), p4 = cvPoint(u2 + n2, v2 + m2);
		cv::Scalar green = CV_RGB(0, 255, 0);
		cv::rectangle(m, p3, p4, green);
	}
	//cv::imshow("gpu_clean", m);

	const std::string& new_img_path = "results/" + vid_name + '_' + vid_num_str + "_" + info +
		"_frame_" + frame_num_str + ".jpg";
	imwrite(new_img_path, m);
}

float get_gray(int x, int y, cv::Mat img) {
	cv::Vec3b c = img.at<cv::Vec3b>(cv::Point(x, y));
	float r = c.val[0];
	float g = c.val[1];
	float b = c.val[2];
	float gr = (r + g + b) / 3.0f;
	return gr;
}

float get_color(int x, int y, cv::Mat img, int rgb) {
	cv::Vec3b c = img.at<cv::Vec3b>(cv::Point(x, y));
	return c.val[rgb];
}

// Filtro Sobel
float sobel(cv::Mat frame, int u, int v, int rgb) {
	double Gx, Gy, res, frac, ent;
	
	Gx = get_color(u-1, v+1, frame, rgb) + 
		2 * get_color(u, v+1, frame, rgb) +
		get_color(u+1, v+1, frame, rgb) -
		get_color(u-1, v-1, frame, rgb) -
		2 * get_color(u, v-1, frame, rgb) -
		get_color(u+1, v-1, frame, rgb);

	Gy = get_color(u+1, v-1, frame, rgb) +
		2 * get_color(u+1, v, frame, rgb) +
		get_color(u+1, v+1, frame, rgb) -
		get_color(u-1, v-1, frame, rgb) -
		2 * get_color(u-1, v, frame, rgb) -
		get_color(u-1, v+1, frame, rgb);

	res = sqrt(Gx * Gx + Gy * Gy);

	ent = trunc(res);
	frac = res - ent;
	res = ent;
	if ((res >= 0) && (frac > 0.5))
		res++;

	return res;
}

float sobel_rgb(int x, int y, cv::Mat img) {
	double r = sobel(img, x, y, 0);
	double g = sobel(img, x, y, 1);
	double b = sobel(img, x, y, 2);
	return r + g + b;
}

// Calculo de IoU
double intersectionOverUnion () {

	//Obtener truth
	short u1 = u0, v1 = v0;
	short n1 = w0, m1 = h0;

	double n3 = std::max(0, std::min(u1 + n1, u2 + n2) - std::max(u1, u2));
	double m3 = std::max(0, std::min(v1 + m1, v2 + m2) - std::max(v1, v2));

	double Ai = n3*m3;

	double Au = ((n1*m1) + (n2*m2)) - Ai;

	return (Ai/Au);
}

float fscore(int Fa, int Fb) {
	float fdiv = (Fb/2);
	float sum = Fa + fdiv;
	return(Fa/sum);
}

void calculateTTime(double tseconds) {
	int min = floor(tseconds/60);
	int seg = round(tseconds - (60*min));
	printf("\nTotal time: %d mins. %d segs.\n", min, seg);
}

void calculateFPS(double tseconds, int frameI, int frameF) {
	int tframe = frameF - frameI;
	double fps = tframe/tseconds;
	printf("\nFrames per second: %f\n", fps);
}

void writeIoU (int frame, double iou) {
	std::ofstream iou_result_file;
	iou_result_file.open(iou_route.c_str(), std::ofstream::app);

	iou_result_file << frame << "," << iou << "\n";
	iou_result_file.close();
}

// Función principal
int main(int argc, char** argv) {
	struct timeval bigBegin;
	struct timeval bigEnd;
	cv::Point p1, p2, p3, p4;
	cv::Scalar red = CV_RGB(255, 0, 0), green = CV_RGB(0, 255, 0);

	gettimeofday(&bigBegin, NULL);

	// Lectura del archivo de configuración
	read_config();

	// Lectura del archivo del kernel
	if (honeybee == 0)
		kernel_size = read_kernel(kernel_file, &kernel_src);
	else
		kernel_size = read_kernel(kernel_file_bee, &kernel_src);

	// Lectura del archivo de anotación
	read_ann();

	// Par de frames a comparar
	cv::Mat frame1;
	cv::Mat frame2;

	// Escritura del archivo de resultados
	result_file.open(path_result.c_str(), std::ofstream::app);
	result_file << "\n" << vid_name << " " << vid_num << "\n";
	if (honeybee == 1) {
		result_file << " - HSA";
	}
	result_file << "\n";
	// Primer frame
	result_file << initial_frame << "," << u0 << "," << v0 << "," << w0 << "," << h0 << "\n";
	result_file.close();

	double avg_time = 0;
	
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	const std::string& ann_path = path_alov + ann_dir + vid_name + '/' +
		vid_name + "_video" + vid_num_str + ".ann";
	FILE* ann_file = fopen(ann_path.c_str(), "r");

	short target_frame, u1, v1, n1, m1;
	double ax, ay, bx, by, cx, cy, dx, dy, iou;
	int Fa = 0, Fb = 0;

	// Preparación del archivo de anotación para obtener el valor ground truth (con el fin de obtener el valor F-Score)
	if (take_metrics == 1) {

		fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
			&target_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);
		fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
			&target_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);

		extractIoUInfo(ax, ay, bx, by, cx, cy, dx, dy);

		std::ofstream iou_result_file;
		iou_result_file.open(iou_route.c_str(), std::ofstream::app);
		if(use_sobel)
			check_str = " Sobel";
		else
			check_str = "";
		if(resting)
			check_str = check_str + " Resting";
		if(honeybee)
			check_str = check_str + " HSA";
		if(resizea)
			check_str = check_str + " Resize";
		iou_result_file << "\n" << vid_name << "_0" << vid_num << check_str << "\n";
		iou_result_file.close();
	}

	// Ciclo principal que procesa cada frame
	for (int current = initial_frame + 1; current <= last_frame; current++) {

		printf("\nCurrent frame: %d of %d\n", current, last_frame);

		struct timeval begin;
		struct timeval end;

		// Preparación de la GPU
		set_gpu();

		// Lectura de imágenes
		frame1 = read_frame(current - 1);
		frame2 = read_frame(current);

		gettimeofday(&begin, NULL);

		if (window > max_window && resizea == 1) {
			img_ratio = (double)window / (double)max_window;

			// Re-escalar
			cv::Mat temp;
			cv::Mat temp2;
			cv::Size size(frame1.cols / img_ratio, frame1.rows / img_ratio);
			cv::resize(frame1, temp, size);
			cv::resize(frame2, temp2, size);

			// Clonar
			frame1 = temp.clone();
			frame2 = temp2.clone();
		}

		// Obtención de NCC
		int mx = frame1.cols;
		int my = frame1.rows;
		int nx = w0;
		int ny = h0;
		float * gamma = 
			(float*)std::malloc(((mx - nx) * (my - ny)) * sizeof(float));
		for (int x = 0; x < (mx - nx); x++)
			for (int y = 0; y < (my - ny); y++)
				gamma[x + y * (mx - nx)] = 0.0f;

		// Números aleatorios pre-generados
		unsigned int *rand1 = (unsigned int*)std::malloc((mx * my) * sizeof(int));
		float *rand2 = (float*)std::malloc((mx * my) * sizeof(float));
		float *rand3 = (float*)std::malloc((mx * my) * sizeof(float));
		float *rand4 = (float*)std::malloc((mx * my) * sizeof(float));

		// Generación de nuevos números aleatorios
		for (int i = 0; i < mx * my; i++) {
			rand1[i] = cvRandInt(&rng);
			rand2[i] = cvRandReal(&rng);
			rand3[i] = get_beta(rand2[i], eta_c);
			rand4[i] = get_delta(rand2[i], eta_m);
		}

		// Filtros
		float * frame1_gray = 
			(float*)std::malloc((frame1.cols * frame1.rows) * sizeof(float));
		float * frame2_gray = 
			(float*)std::malloc((frame2.cols * frame2.rows) * sizeof(float));
		for (int x = 0; x < frame1.cols; x++) {
			for (int y = 0; y < frame1.rows; y++) {
				if (use_sobel == 0)
					frame1_gray[x + y * frame1.cols] = get_gray(x, y, frame1);
				else {
					frame1_gray[x + y * frame1.cols] = sobel_rgb(x, y, frame1);
				}
			}
		}
		for (int x = 0; x < frame2.cols; x++) {
			for (int y = 0; y < frame2.rows; y++) {
				if (use_sobel == 0)
					frame2_gray[x + y * frame2.cols] = get_gray(x, y, frame2);
				else {
					frame2_gray[x + y * frame2.cols] = sobel_rgb(x, y, frame2);
				}
			}
		}

		// AMD recomienda múltiplos de 64
		if (honeybee == 0) {
			localWorkSize[0] = 16;
			localWorkSize[1] = 16;
		} 
		else {
			localWorkSize[0] = max_work_group_size;
			localWorkSize[1] = 1;
		}

		// Solo un grupo local para permitir la coordinación
		if (honeybee == 0) {
			globalWorkSize[0] = (int)(localWorkSize[0] *
				ceil((mx - nx) / (float)(localWorkSize[0])));
			globalWorkSize[1] = (int)(localWorkSize[0] *
				ceil((my - ny) / (float)(localWorkSize[0])));
		} 
		else {
			// Solo un grupo local para permitir la coordinación
			globalWorkSize[0] = localWorkSize[0];
			globalWorkSize[1] = localWorkSize[1];
		}

		// Dispositivo de memoria OpenCL
		cl_mem d_frame1, d_frame2, d_gamma, d_rand1, d_rand2, d_rand3, d_rand4,
			d_mu_e_bees, d_mu_e_obj, d_lambda_e_bees, d_lambda_e_obj,
			d_mu_lambda_bees, d_mu_lambda_obj, d_mu_lambda_order,
			d_mu_r_bees, d_mu_r_obj, d_lambda_r_bees, d_lambda_r_obj, 
			d_temp;

		// Asignar memoria de GPU
		d_frame1 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
			CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), frame1_gray, &err);
		d_frame2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), frame2_gray, &err);
		d_gamma = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, ((mx - nx) * (my - ny)) * sizeof(float), gamma, &err);
		if (honeybee == 1) {
			// Números aleatorios
			d_rand1 = clCreateBuffer(context, CL_MEM_READ_WRITE |
				CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(int), rand1, &err);
			d_rand2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
				CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), rand2, &err);
			d_rand3 = clCreateBuffer(context, CL_MEM_READ_WRITE |
				CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), rand3, &err);
			d_rand4 = clCreateBuffer(context, CL_MEM_READ_WRITE |
				CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), rand4, &err);

			// Abejas
			d_mu_e_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float), NULL, &err);
			d_mu_e_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * sizeof(float), NULL, &err);
			d_lambda_e_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float), NULL, &err);
			d_lambda_e_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * sizeof(float), NULL, &err);

			// Mu + valor lambda de abejas, tamaño doble para emplear el método "merge sort"
			d_mu_lambda_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float) * 2 * 2, NULL, &err);
			d_mu_lambda_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float) * 2, NULL, &err);
			d_mu_lambda_order = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float) * 2, NULL, &err);

			// Abejas
			d_mu_r_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float), NULL, &err);
			d_mu_r_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * sizeof(float), NULL, &err);
			d_lambda_r_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * 2 * sizeof(float), NULL, &err);
			d_lambda_r_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				num_bees * sizeof(float), NULL, &err);

			d_temp = clCreateBuffer(context, CL_MEM_READ_WRITE,
				2 * num_bees * sizeof(short), NULL, &err);
		}
		// Revisar la correcta asignación de memoria
		if (!d_frame1 || !d_frame2 || !d_gamma) {
			printf("Error: Failed to allocate device memory!\n");
			exit(1);
		}

		// Ejecutar kernel OpenCL
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_frame1);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_frame2);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_gamma);
		err |= clSetKernelArg(kernel, 3, sizeof(short), (short *)&u0);
		err |= clSetKernelArg(kernel, 4, sizeof(short), (short *)&v0);
		err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&nx);
		err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&ny);
		err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&mx);
		err |= clSetKernelArg(kernel, 8, sizeof(int), (void *)&my);
		if (honeybee == 1) {
			err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&d_rand1);
			err |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&d_rand2);
			err |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&d_rand3);
			err |= clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&d_rand4);

			err |= clSetKernelArg(kernel, 13, sizeof(short), (void *)&max_gen);
			err |= clSetKernelArg(kernel, 14, sizeof(float), (void *)&rate_beta_e);
			err |= clSetKernelArg(kernel, 15, sizeof(float), (void *)&rate_alpha_e);
			err |= clSetKernelArg(kernel, 16, sizeof(float), (void *)&rate_gamma_e);
			err |= clSetKernelArg(kernel, 17, sizeof(float), (void *)&rate_beta_r);
			err |= clSetKernelArg(kernel, 18, sizeof(float), (void *)&rate_alpha_r);
			err |= clSetKernelArg(kernel, 19, sizeof(float), (void *)&rate_gamma_r);

			err |= clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *)&d_mu_e_bees);
			err |= clSetKernelArg(kernel, 21, sizeof(cl_mem), (void *)&d_mu_e_obj);
			err |= clSetKernelArg(kernel, 22, sizeof(cl_mem), (void *)&d_lambda_e_bees);
			err |= clSetKernelArg(kernel, 23, sizeof(cl_mem), (void *)&d_lambda_e_obj);

			err |= clSetKernelArg(kernel, 24, sizeof(cl_mem), (void *)&d_mu_lambda_bees);
			err |= clSetKernelArg(kernel, 25, sizeof(cl_mem), (void *)&d_mu_lambda_obj);
			err |= clSetKernelArg(kernel, 26, sizeof(cl_mem), (void *)&d_mu_lambda_order);

			err |= clSetKernelArg(kernel, 27, sizeof(cl_mem), (void *)&d_mu_r_bees);
			err |= clSetKernelArg(kernel, 28, sizeof(cl_mem), (void *)&d_mu_r_obj);
			err |= clSetKernelArg(kernel, 29, sizeof(cl_mem), (void *)&d_lambda_r_bees);
			err |= clSetKernelArg(kernel, 30, sizeof(cl_mem), (void *)&d_lambda_r_obj);

			err |= clSetKernelArg(kernel, 31, sizeof(cl_mem), (void *)&d_temp);
		}
		if (err != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments! %d\n", err);
			exit(1);
		}
		err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, 
			globalWorkSize, localWorkSize, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(1);
		}
		clFinish(commands);

		// Recuperar resultados del dispositivo
		err = clEnqueueReadBuffer(commands, d_gamma, CL_TRUE, 0, 
			(mx - nx) * (my - ny) * sizeof(float), gamma, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to read output array! %d\n", err);
			exit(1);
		}

		// Encontrar el mejor valor gamma
		float maxGamma = gamma[0];
		int maxI = 0;
		int todos = 0;
		for (int i = 1; i < (mx - nx) * (my - ny); i++) {
			if (gamma[i] > maxGamma) {
				maxGamma = gamma[i];
				maxI = i;
				if (gamma[i] != -100.00f)
					todos = 1;
			}
		}
		if (todos == 1)
			printf("Valid result: %f\n", maxGamma);

		// Nuevos u0, v0
		u0 = maxI % (mx - nx);
		v0 = maxI / (mx - nx);

		// Liberar memoria
		clReleaseMemObject(d_frame1);
		clReleaseMemObject(d_frame2);
		clReleaseMemObject(d_gamma);
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(commands);
		clReleaseContext(context);
		free(frame1_gray);
		free(frame2_gray);
		free(gamma);
		if (honeybee == 1) {
			free(rand1);
			free(rand2);
			free(rand3);
			free(rand4);
			clReleaseMemObject(d_rand1);
			clReleaseMemObject(d_rand2);
			clReleaseMemObject(d_rand3);
			clReleaseMemObject(d_rand4);
			clReleaseMemObject(d_mu_e_bees);
			clReleaseMemObject(d_mu_e_obj);
			clReleaseMemObject(d_lambda_e_bees);
			clReleaseMemObject(d_lambda_e_obj);
			clReleaseMemObject(d_mu_lambda_bees);
			clReleaseMemObject(d_mu_lambda_obj);
			clReleaseMemObject(d_mu_lambda_order);
			clReleaseMemObject(d_mu_r_bees);
			clReleaseMemObject(d_mu_r_obj);
			clReleaseMemObject(d_lambda_r_bees);
			clReleaseMemObject(d_lambda_r_obj);
			clReleaseMemObject(d_temp);
		}

		gettimeofday(&end, NULL);
		long int beginl = begin.tv_sec * 1000 + begin.tv_usec / 1000;
		long int endl = end.tv_sec * 1000 + end.tv_usec / 1000;
		double elapsed_secs = double(endl - beginl) / 1000;
		printf("Elapsed seconds: %f\n", elapsed_secs);

		result_file.open(path_result.c_str(), std::ofstream::app);
		avg_time += elapsed_secs;
		result_file << current << "," << u0 
			<< "," << v0 << "," << w0 << "," << h0 << "\n";
		result_file.close();

		if (take_metrics == 1) {
			if(current == target_frame) {
				iou = intersectionOverUnion();
				printf("IoU: %f\n\n", iou);
				if(iou >= 0.5)
					Fa++;
				else
					Fb++;

				// Guardar imágenes
				if (save_track == 1)
					save_img(frame2, current, check_str, 1);

				fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
					&target_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);
				extractIoUInfo(ax, ay, bx, by, cx, cy, dx, dy);
				writeIoU(current, iou);
			}
			else {
				printf("\n");

				// Guardar imágenes
				if (save_track == 1)
					save_img(frame2, current, check_str, 0);
			}
		}

		// Descanso cada 100 frames
		if ((current - initial_frame) % 100 == 0 && resting == 1) {
			if (avg_time /= current - initial_frame > 1.5) {
				printf("Resting...\n");
				gettimeofday(&begin, NULL);
				while (true) {
					gettimeofday(&end, NULL);
					long int beginl = begin.tv_sec * 1000 + begin.tv_usec / 1000;
					long int endl = end.tv_sec * 1000 + end.tv_usec / 1000;
					double elapsed_secs = double(endl - beginl) / 1000;
					if (elapsed_secs > 180)
						break;
				}
			}
		}
	}

	// Cerrar archivo de anotación
	if (take_metrics == 1)
		fclose(ann_file);

	avg_time /= last_frame - (initial_frame + 1);
	printf("Average seconds: %f\n", avg_time);

	gettimeofday(&bigEnd, NULL);
	long int bigBeginl = bigBegin.tv_sec * 1000 + bigBegin.tv_usec / 1000;
	long int bigEndl = bigEnd.tv_sec * 1000 + bigEnd.tv_usec / 1000;
	double tseconds = double(bigEndl - bigBeginl) / 1000;
	calculateTTime(tseconds);
	if (take_metrics == 1)
		calculateFPS(tseconds, initial_frame, last_frame);

	/*
		Truth:
			u1: u0
			v1: v0
			n1: w0
			m1: h0

		Ground Truth:
			u2: u2
			v2: v2
			n2: n2
			m2: m2
	*/


	if (take_metrics == 1) {
		printf("\nLast frame's IoU: %f\n", iou);
		printf("F-Score: %f\n", fscore(Fa, Fb));
	}

	return 0;
}