// Obtener un entero aleatorio del arreglo pre-generado
unsigned int random_int(unsigned int *rand_index, int rand_size, __global unsigned int* rand1) {
	unsigned int r = rand1[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Obtener un número de tipo doble aleatorio del arreglo pre-generado
float random_double(unsigned int *rand_index, int rand_size, __global float* rand2) {
	float r = rand2[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Obtener un valor delta aleatorio para la mutación polinomial del arreglo pre-generado
float random_delta(unsigned int *rand_index, int rand_size, __global float* rand4) {
	float r = rand4[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Obtener un valor beta aleatorio para el Cruce Binario Simulado (SBX) del arreglo pre-generado
float random_beta(unsigned int *rand_index, int rand_size, __global float* rand3) {
	float r = rand3[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Distancia euclidiana
float ec_distance (float x1, float y1, float x2, float y2) {
	float a = x1 - x2;
	float b = y1 - y2;
	float d = sqrt(a * a + b * b);
	return d;
}

// Generar una población aleatoria inicial
void initial_random_pop(
	unsigned int *rand_index,
	int rand_size,
	__global float* rand2,
	__global float* mu_bees,
	int n,
	int bee,
	float* limits) {

	// Dos componentes por abeja
	// Primer componente, primer núcleo
	if (n == 0) {
		float a = random_double(rand_index, rand_size, rand2);
		float up = limits[0];
		float low = limits[1];
		if (up > limits[2])
			up = limits[2];
		if (low < limits[3])
			low = limits[3];
		mu_bees[bee * 2] = (a * (up - low)) + low;
	}

	// Segundo componente, segundo núcleo
	if (n == 1) {
		float a = random_double(rand_index, rand_size, rand2);
		float up = limits[4];
		float low = limits[5];
		if (up > limits[6])
			up = limits[6];
		if (low < limits[7])
			low = limits[7];
		mu_bees[bee * 2 + 1] = (a * (up - low)) + low;
	}

	// Esperar por todos los kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// ZNCC: comparación de frames
float zncc(
	int u1,
	int v1,
	int u2,
	int v2,
	__global unsigned char* frame1,
	__global unsigned char* frame2,
	int maxX,
	int maxY,
	int nx,
	int ny,
	int n,
	int bee,
	int global_id,
	__global float* gamma,
	__global float* res,
	int cores_per_bee) {

	int i, k, l, stepx, stepy, ini, fin;
	float sumnum, sumden1, sumden2, my_int_F1, my_int_F2, int_F1, int_F2;
	int mx = maxX;

	my_int_F1 = 0.0f;
	my_int_F2 = 0.0f;
	
	ini = ((nx * ny) / 4) * n;
	fin = (((nx * ny) / 4) * (n + 1)) -1;
	if (n == 3) {
		fin += (nx * ny) - (((nx * ny) / 4) * 4);
	}
	
	for (i = ini; i <= fin; i++) {
		l = i / nx;
		k = i % nx;

		int_F1 = frame1[(u1 + l) + (v1 + k) * mx];
		int_F2 = frame2[(u2 + l) + (v2 + k) * mx];
		my_int_F1 += int_F1;
		my_int_F2 += int_F2;
	}

	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// Guardar resultado en memoria global acumulada
	gamma[global_id * 2] = my_int_F1;
	gamma[global_id * 2 + 1] = my_int_F2;
	
	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	sumnum = 0.0;
	sumden1 = 0.0;
	sumden2 = 0.0;
	if (n == 0) {
		my_int_F1 = 0.0f;
		my_int_F2 = 0.0f;
		
		// Obtener acumulación de todos los núcleos
		for (int i = 0; i < cores_per_bee; i++) {
			my_int_F1 += gamma[(global_id + i) * 2];
			my_int_F2 += gamma[(global_id + i) * 2 + 1];
		}
		
		my_int_F1 = my_int_F1 / (nx * ny);
		my_int_F2 = my_int_F2 / (nx * ny);
		
		// Guardar resultado en la memoria global para que todos los núcleos puedan usarlo
		gamma[global_id * 2] = my_int_F1;
		gamma[global_id * 2 + 1] = my_int_F2;
	}

	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// Copiar resultados desde la memoria global
	my_int_F1 = gamma[(global_id - n) * 2];
	my_int_F2 = gamma[(global_id - n) * 2 + 1];		
	
	for (i = ini; i <= fin; i++) {
		l = i / nx;
		k = i % nx;

		int_F1 = frame1[(u1 + l) + (v1 + k) * mx];
		int_F2 = frame2[(u2 + l) + (v2 + k) * mx];
		sumnum += (int_F1 - my_int_F1) * (int_F2 - my_int_F2);
		sumden1 += (int_F1 - my_int_F1) * (int_F1 - my_int_F1);
		sumden2 += (int_F2 - my_int_F2) * (int_F2 - my_int_F2);
	}

	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// Guardar resultado en memoria global acumulada
	gamma[global_id * 3] = sumnum;
	gamma[global_id * 3 + 1] = sumden1;
	gamma[global_id * 3 + 2] = sumden2;
	
	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if (n == 0) {
		sumnum = 0;
		sumden1 = 0;
		sumden2 = 0;
		
		// Obtener acumulación de todos los núcleos
		for (int i = 0; i < cores_per_bee; i++) {
			sumnum += gamma[(global_id + i) * 3];
			sumden1 += gamma[(global_id + i) * 3 + 1];
			sumden2 += gamma[(global_id + i) * 3 + 2];
		}
		
		res[bee] = (sumnum / (sqrt(sumden1 * sumden2)));
	}

	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Limpiar memoria global
	gamma[global_id * 3] = 0.0f;
	gamma[global_id * 3 + 1] = 0.0f;
	gamma[global_id * 3 + 2] = 0.0f;
	
	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	return 0.0f;
}

// Función de aptitud
float fitness_function(
	int u,
	int v,
	int a,
	int b,
	int maxX,
	int maxY,
	__global unsigned char* frame1,
	__global unsigned char* frame2,
	int nx,
	int ny,
	int n,
	int bee,
	int global_id,
	__global float* gamma,
	__global float* res,
	int cores_per_bee) {

	float f_12, F;
	
	// Limpieza de memoria
	res[bee] = 0.0f;
	
	// Esperar por todos los núcleos
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Ejecutar ZNCC
	float zn = zncc(u, v, a, b, frame1, frame2, maxX, maxY, nx, ny, n, bee, global_id,
		gamma, res, cores_per_bee);

	// Regresar salida
	return 0.0f;
}

// Evaluar población
void eval_pop(
	__global float* bees,
	__global float* obj,
	int n,
	int bee,
	int global_id,
	int mx,
	int my,
	__global unsigned char* frame1,
	__global unsigned char* frame2,
	__global float* gamma,
	int u,
	int v,
	int nx,
	int ny,
	float* limits,
	int cores_per_bee) {
	
	// Punto en frame 2
	int a = bees[bee * 2];
	int b = bees[bee * 2 + 1];
	
	// Revisar limites, solo por si acaso
	if (a > limits[2])
		a = limits[2];
	if (a < limits[3])
		a = limits[3];
	if (b > limits[6])
		b = limits[6];
	if (b < limits[7])
		b = limits[7];

	// Obtener valor de aptitud del punto
	float ff = fitness_function(u, v, a, b, mx, my, frame1, frame2, nx, ny, n, bee, 
		global_id, gamma, obj, cores_per_bee);

	// Esperar por todos los núcleos
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Mutación polinomial de un individuo (abeja) dado
void mutation(
	__global float* bees,
	int bee,
	unsigned int *rand_index,
	int rand_size,
	__global float* rand4,
	float* limits) {

	float x, delta;
	int site;

	// Mutación por cada componente del individuo
	for (site = 0; site < 2; site++) {
		// Obtener limites, basándose en el componente
		float upper = limits[site * 4 + 2];
		float lower = limits[site * 4 + 3];

		// Obtener el valor
		x = bees[bee * 2 + site];

		// Obtener delta
		delta = random_delta(rand_index, rand_size, rand4);

		// Aplicar mutación
		if (delta >= 0) {
			bees[bee * 2 + site] += delta * (upper - x);
		} else {
			bees[bee * 2 + site] += delta * (x - lower);
		}

		// Limites absolutos
		if (bees[bee * 2 + site] < lower)
			bees[bee * 2 + site] = lower;
		if (bees[bee * 2 + site] > upper)
			bees[bee * 2 + site] = upper;
	}
}

// Cruce Binario Simulado (SBX)
void create_children(
	float p1,
	float p2,
	__global float *c1,
	__global float *c2,
	float low,
	float high,
	unsigned int *rand_index,
	int rand_size,
	__global float* rand3) {

	float beta = random_beta(rand_index, rand_size, rand3);

	float v2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2);
	float v1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2);

	if (v2 < low) v2 = low;
	if (v2 > high) v2 = high;
	if (v1 < low) v1 = low;
	if (v1 > high) v1 = high;

	*c2 = v2;
	*c1 = v1;
}

// Cruce de dos padres para generar dos hijos
void cross_over(
	int parent1,
	int parent2,
	int child1,
	int child2,
	__global float* mu_bees,
	__global float* lambda_bees,
	unsigned int *rand_index,
	int rand_size,
	__global float* rand3,
	float* limits) {

	int site;
	int nvar_real = 2;
	for (site = 0; site < nvar_real; site++) {
		float lower = limits[site * 4 + 3];
		float upper = limits[site * 4 + 2];

		create_children(
			mu_bees[parent1 * nvar_real + site],
			mu_bees[parent2 * nvar_real + site],
			&lambda_bees[child1 * nvar_real + site],
			&lambda_bees[child2 * nvar_real + site],
			lower,
			upper,
			rand_index,
			rand_size,
			rand3);
	}
}

// Generar nueva población usando selección, cruce, mutación y búsqueda aleatoria
void generate_new_pop(
	__global unsigned int* rand1,
	unsigned int *rand_index,
	int rand_size,
	int n,
	float rate_alpha,
	float rate_beta,
	float rate_gamma,
	__global float* mu_obj,
	__global float* mu_bees,
	__global float* lambda_bees,
	__global float* rand2,
	__global float* rand3,
	__global float* rand4,
	int bee,
	float* limits,
	int num_bees) {

	int mate1, mate2, num_cross, num_mut, num_rand;

	// Truncar niveles de mutación, cruce y búsqueda aleatoria
	int rate_mut = rate_alpha;
	int rate_cross = rate_beta;
	int rate_rand = rate_gamma;

	// Mutación
	// abejas desde 0 hasta rate_mut - 1
	// Solo el núcleo 0 de cada abeja
	if (bee >= 0 && bee <= rate_mut - 1 && n == 0) {
		// Selección
		int a = random_int(rand_index, rand_size, rand1) % num_bees;
		int b = random_int(rand_index, rand_size, rand1) % num_bees;
		if (mu_obj[a] > mu_obj[b])
			mate1 = a;
		else
			mate1 = b;

		// Copiar al individuo
		lambda_bees[bee * 2] = mu_bees[mate1 * 2];
		lambda_bees[bee * 2 + 1] = mu_bees[mate1 * 2 + 1];

		// Mutación polinomial
		mutation(lambda_bees, bee, rand_index, rand_size, rand4, limits);
	}

	// Cruce
	// abejas desde first_bee + rate_mut hasta first_bee + rate_mut + rate_cross - 1
	// rate_mut debe de seguir existiendo aun si el cruce pasa ya que dos hijos son generados
	if (bee >= rate_mut && 
		bee <= rate_mut + rate_cross - 1 &&
		n == 0 &&
		bee % 2 == 0) {
	
		// Selección
		int a = random_int(rand_index, rand_size, rand1) % num_bees;
		int b = random_int(rand_index, rand_size, rand1) % num_bees;
		int c = random_int(rand_index, rand_size, rand1) % num_bees;
		int d = random_int(rand_index, rand_size, rand1) % num_bees;
		if (mu_obj[a] > mu_obj[b])
			mate1 = a;
		else
			mate1 = b;
		if (mu_obj[c] > mu_obj[d])
			mate2 = c;
		else
			mate2 = d;

		// Cruce SBX
		cross_over(mate1, mate2, bee, bee + 1, mu_bees, lambda_bees,
			rand_index, rand_size, rand3, limits);
	}

	// Selección aleatoria
	// abejas desde first_bee + rate_mut + rate_cross hasta first_bee + rate_mut + rate_cross + rate_rand - 1
	if (bee >= rate_mut + rate_cross
		&& bee <= rate_mut + rate_cross + rate_rand - 1
		&& n == 0) {

		int nvar_real = 2;
		float lower;
		float upper;
		for (int j = 0; j < nvar_real; j++) {
			upper = limits[j * 4 + 2];
			lower = limits[j * 4 + 3];

			lambda_bees[bee * nvar_real + j] =
				(float)(random_double(rand_index, rand_size, rand2) * 
				(upper - lower) + lower);
		}
	}

	// Esperar por todos los kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Mezclar poblaciones mu y lambda
void merge_pop(
	__global float* mu_obj,
	__global float* lambda_obj,
	__global float* mu_lambda_obj,
	__global float* mu_lambda_bees,
	__global float* mu_bees,
	__global float* lambda_bees,
	int bee,
	int n,
	__global float* mu_lambda_order) {
	
	if (n == 0) {
		// Copy mu bee
		int mu_lambda_bee = bee * 2;
		mu_lambda_obj[mu_lambda_bee] = mu_obj[bee];
		mu_lambda_bees[mu_lambda_bee * 2] = mu_bees[bee * 2];
		mu_lambda_bees[mu_lambda_bee * 2 + 1] = mu_bees[bee * 2 + 1];
		mu_lambda_order[mu_lambda_bee] = mu_lambda_bee;

		// Copiar abejas lambda
		mu_lambda_bee++;
		mu_lambda_obj[mu_lambda_bee] = lambda_obj[bee];
		mu_lambda_bees[mu_lambda_bee * 2] = lambda_bees[bee * 2];
		mu_lambda_bees[mu_lambda_bee * 2 + 1] = lambda_bees[bee * 2 + 1];
		mu_lambda_order[mu_lambda_bee] = mu_lambda_bee;
	}

	// Esperar por todos los kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Algoritmo "Insertion sort" (ordenamiento por inserción) para ordenar la lista local priorizando el valor objetivo
// n debe ser un valor pequeño
void insertion_sort(
	__global float* numbers,
	__global float* order,
	int n,
	int list_id,
	float sign) {

	int first_num = list_id * n;
	
	for (int i = 1; i < n; i++) {
		float x = numbers[first_num + i];
		float o = order[first_num + i];

		int j = i - 1;
		while (j >= 0 && numbers[first_num + j] * sign > x * sign) {
			numbers[first_num + j + 1] = numbers[first_num + j];
			order[first_num + j + 1] = order[first_num + j];
			j--;
		}
		numbers[first_num + j + 1] = x;
		order[first_num + j + 1] = o;
	}
}

// Función del algoritmo "Merge Sort"
void merge_sort(
	__global float* numbers,
	__global float* order,
	int n,
	int sublist_id,
	int num_sublists,
	float sign,
	int max_sublists) {

	// El primer número de ambas sublistas
	int first_num = sublist_id * n;
	int first_num2 = (sublist_id + num_sublists / 2) * n;

	// El primer número de la lista temporal
	int first_temp = n * max_sublists;

	int j = 0;
	int k = 0;
	for (int i = 0; i < n * num_sublists; i++) {
		// Si ambas listas tienen números restantes revisar cual es el más pequeño
		if (j < (n * num_sublists) / 2 && k < (n * num_sublists) / 2) {
			// Ordenar
			if (numbers[first_num + j] * sign < numbers[first_num2 + k] * sign) {
				numbers[first_temp + first_num + i] =
					numbers[first_num + j];
				order[first_temp + first_num + i] =
					order[first_num + j];
				j++;
			} else {
				numbers[first_temp + first_num + i] =
					numbers[first_num2 + k];
				order[first_temp + first_num + i] =
					order[first_num2 + k];
				k++;
			}
		// Si solo una sublista tiene números restantes, copiar todos
		} else {
			if (j < k) {
				numbers[first_temp + first_num + i] =
					numbers[first_num + j];
				order[first_temp + first_num + i] =
					order[first_num + j];
				j++;
			} else if (k < j) {
				numbers[first_temp + first_num + i] =
					numbers[first_num2 + k];
				order[first_temp + first_num + i] =
					order[first_num2 + k];
				k++;
			}
		}
	}

	// Copiar orden del arreglo temporal al arreglo original
	for (int i = 0; i < n * num_sublists; i++) {
		numbers[first_num + i] = numbers[first_temp + first_num + i];
		order[first_num + i] = order[first_temp + first_num + i];
	}
}

// Obtener el mejor individuo mu
void best_mu(
	__global float* mu_lambda_obj,
	__global float* mu_lambda_order,
	int bee,
	__global float* mu_bees,
	__global float* mu_obj,
	__global float* mu_lambda_bees,
	int n,
	int num_bees) {

	// Ordenar lista local
	if (n == 0) {
		insertion_sort(mu_lambda_obj, mu_lambda_order, 2, bee, -1);
	}

	// Esperar por todos los kernels
	barrier(CLK_LOCAL_MEM_FENCE);

	int max_sublists = num_bees;
	// Ordenar lista global
	for (int num_sublists = 2; num_sublists <= max_sublists; num_sublists *= 2) {
		if (bee % num_sublists == 0 && n == 0) {
			merge_sort(mu_lambda_obj, mu_lambda_order, 2, bee, num_sublists,
				-1, max_sublists);
		}
		// Esperar por todos los kernels
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (n == 0) {
		// Copiar la mejor abeja mu
		// El índice actual es la mejor abeja
		int o = mu_lambda_order[bee];

		// Copiar la mejor abeja a la población mu
		mu_obj[bee] = mu_lambda_obj[bee];
		mu_bees[bee * 2] = mu_lambda_bees[o * 2];
		mu_bees[bee * 2 + 1] = mu_lambda_bees[o * 2 + 1];
	}

	// Esperar por todos los kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Dibujar abejas
void draw(
	__global float* bees,
	__global float* obj,
	int bee,
	int n,
	__global float* gamma,
	int mx,
	int nx) {

	// El primer núcleo dibuja las abejas
	if (n == 0) {
		int x = bees[bee * 2];
		int y = bees[bee * 2 + 1];
		gamma[x + y * (mx - nx)] = obj[bee];
	}

	// Esperar por todos los kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Kernel OpenCL
__kernel void fitness(
	__global float* frame1, // Primer frame del video
	__global float* frame2, // Segundo frame del video
	__global float* gamma, // Frame usado para desplegar resultados
	short u0,
	short v0,
	int nx,
	int ny,
	int mx,
	int my,

	__global unsigned int* rand1, // enteros aleatorios
	__global float* rand2, // valores de tipo doble aleatorios
	__global float* rand3, // valor beta aleatorio para cruce SBX
	__global float* rand4, // valor delta aleatorio para mutación polinomial

	short max_gen, // Número de generaciones de abejas
	float rate_beta_e, // Porcentaje de hijos cruce para exploración
	float rate_alpha_e, // Porcentaje de hijos mutación para exploración
	float rate_gamma_e, // Porcentaje de hijos aleatorios para exploración
	float rate_beta_r, // Porcentaje de hijos cruce para recolección
	float rate_alpha_r, // Porcentaje de hijos mutación para recolección
	float rate_gamma_r, // Porcentaje de hijos aleatorios para recolección

	__global float* mu_e_bees, // Valores verdaderos para cada abeja exploradora padre
	__global float* mu_e_obj, // Valor objetivo para cada abeja exploradora padre

	__global float* lambda_e_bees, // Valores verdaderos para cada abeja exploradora hija
	__global float* lambda_e_obj, // Valor objetivo para cada abeja exploradora hija

	__global float* mu_lambda_bees, // Valores verdaderos para cada abeja exploradora
	__global float* mu_lambda_obj, // Valor objetivo para cada abeja exploradora
	__global float* mu_lambda_order, // Orden de cada abeja exploradora

	__global float* mu_r_bees, // Valores verdaderos para cada abeja recolectora padre
	__global float* mu_r_obj, // Valor objetivo para cada abeja recolectora padre

	__global float* lambda_r_bees, // Valores verdaderos para cada abeja recolectora hija
	__global float* lambda_r_obj, // Valor objetivo para cada abeja recolectora hija

	__global short* recruiter // abejas reclutadoras designadas para cada núcleo, memoria extra es usada
	) {

	// Obtener ID global absoluto y número de núcleos
	int global_id = get_global_id(0) + (get_global_id(1) * get_global_size(0));
	int cores = get_global_size(0) * get_global_size(1);
	
	// Obtener el tamaño de los arreglos aleatorios
	int rand_size = mx * my;

	// Obtener un índice aleatorio, diferentes números aleatorios seguirán diferentes secuencias
	unsigned int rand_index = rand1[global_id % rand_size];

	// Este núcleo pertenece a esta abeja
	int cores_per_bee = 4;
	int bee = global_id / cores_per_bee;

	// Este es el núcleo n-ésimo de la abeja dada
	int n = global_id - (cores_per_bee * bee);

	// Número total de abejas
	int num_bees = cores / cores_per_bee;

	// Límites para cada componente de las abejas
	float limits[8];

	// Primer componente
	int window = nx;
	if (ny > nx)
		window = ny;
	limits[0] = u0 + window / 8;
	limits[1] = u0 - window / 8;
	limits[2] = mx - nx;
	limits[3] = 0;

	// Segundo componente
	limits[4] = v0 + window / 8;
	limits[5] = v0 - window / 8;
	limits[6] = my - ny;
	limits[7] = 0;

	// FASE DE EXPLORACIÓN
	// Generar individuos iniciales aleatorios
	initial_random_pop(&rand_index, rand_size, rand2, mu_e_bees, n, bee, 
		limits);

	for (int generation = 0; generation < max_gen; generation++) {
		// Evaluar población de padres
		eval_pop(mu_e_bees, mu_e_obj, n, bee, global_id, mx, my,
			frame1, frame2, gamma, u0, v0, nx, ny, limits, cores_per_bee);
	
		// Generar población lambda
		generate_new_pop(rand1, &rand_index, rand_size, n,
			rate_alpha_e, rate_beta_e, rate_gamma_e, mu_e_obj, 
			mu_e_bees, lambda_e_bees, rand2, rand3, rand4, 
			bee, limits, num_bees);

		// Evaluar nueva población
		eval_pop(lambda_e_bees, lambda_e_obj, n, bee, global_id, mx, my,
			frame1, frame2, gamma, u0, v0, nx, ny, limits, cores_per_bee);

		// Mu + Lambda
		merge_pop(mu_e_obj, lambda_e_obj, mu_lambda_obj, mu_lambda_bees, mu_e_bees,
			lambda_e_bees, bee, n, mu_lambda_order);

		// Seleccionar el mejor mu
		best_mu(mu_lambda_obj, mu_lambda_order, bee, mu_e_bees, mu_e_obj, 
			mu_lambda_bees, n, num_bees);
	}

	// FASE DE RECLUTAMIENTO
	__global short* recruits = recruiter + num_bees;
	int last_recruiter;
	int min_u;
	int max_u;
	int min_v;
	int max_v;
	if (global_id == 0) {
		float sum = 0.0f;
		int recruited_bees = 0;

		// Obtener valor de aptitud acomulada
		for (int i = 0; i < num_bees; i++) {
			// Valores iguales o menores a 0, no contribuyen
			if (mu_e_obj[i] > 0.0f)
				sum += mu_e_obj[i];
		}

		// Seleccionar recursos de cada abeja reclutadora
		if (sum > 0.0f) {
			for (int i = 0; i < num_bees; i++) {
				// Abejas con valores iguales o menores a 0 no tienen reclutas
				if (mu_e_obj[i] >= 0.0f) {
					recruits[i] = (mu_e_obj[i] / sum) * num_bees;
					recruited_bees += recruits[i];
				} else {
					recruits[i] = 0;
				}
			}
		} else {
			// Ya que las ultimas búsqueda no dieron resultados se realiza una segunda búsqueda normal usando todos los núcleos
			recruits[0] = num_bees;
			mu_e_bees[0] = u0;
			mu_e_bees[1] = v0;
			for (int i = 1; i < num_bees; i++) {
				recruits[i] = 0;
			}
			recruited_bees = num_bees;
		}
		
		// Todos los núcleos deben tener algo de trabajo
		// Dar la diferencia a la mejor abeja exploradora
		if (recruited_bees < num_bees) {
			recruits[0] = recruits[0] + (num_bees - recruited_bees);
		}
		
		// Asignar abejas
		int current_recruiter = 0;
		for (int i = 0; i < num_bees; i++) {
			recruiter[i] = current_recruiter;
			recruits[current_recruiter]--;
			if (recruits[current_recruiter] <= 0)
				current_recruiter++;
		}
		
		// Contar las verdaderas abejas reclutadas
		for (int i = 0; i < num_bees; i++) {
			recruits[i] = 0;
		}
		for (int i = 0; i < num_bees; i++) {
			recruits[recruiter[i]]++;
		}
		
		// Obtener nuevos limites
		min_u = mu_e_bees[0];
		max_u = mu_e_bees[0];
		min_v = mu_e_bees[1];
		max_v = mu_e_bees[1];
		for (int i = 1; i < num_bees; i++) {
			if (mu_e_bees[recruiter[i] * 2] < min_u)
				min_u = mu_e_bees[recruiter[i] * 2];
			if (mu_e_bees[recruiter[i] * 2] > max_u)
				max_u = mu_e_bees[recruiter[i] * 2];
			if (mu_e_bees[recruiter[i] * 2 + 1] < min_v)
				min_v = mu_e_bees[recruiter[i] * 2 + 1];
			if (mu_e_bees[recruiter[i] * 2 + 1] > max_v)
				max_v = mu_e_bees[recruiter[i] * 2 + 1];
		}
		// Salvar en memoria global ya que todos los núcleos necesitan la información
		mu_lambda_obj[0] = min_u;
		mu_lambda_obj[1] = max_u;
		mu_lambda_obj[2] = min_v;
		mu_lambda_obj[3] = max_v;
	}
	// Esperar por todos los kernels
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Guardar en memoria local
	min_u = mu_lambda_obj[0];
	max_u = mu_lambda_obj[1];
	min_v = mu_lambda_obj[2];
	max_v = mu_lambda_obj[3];

	// FASE DE RECOLECCIÓN
	// Nuevos limites basados en el reclutamiento
	// Primer componente
	limits[0] = mu_e_bees[recruiter[bee] * 2] + 1;
	limits[1] = mu_e_bees[recruiter[bee] * 2] - 1;
	limits[2] = max_u;
	limits[3] = min_u;

	// Segundo componente
	limits[4] = mu_e_bees[recruiter[bee] * 2 + 1] + 1;
	limits[5] = mu_e_bees[recruiter[bee] * 2 + 1] - 1;
	limits[6] = max_v;
	limits[7] = min_v;

	// Generar individuos iniciales aleatorios
	initial_random_pop(&rand_index, rand_size, rand2, mu_r_bees, n, bee, 
		limits);
	
	for (int generation = 0; generation < max_gen / 2; generation ++) {
		// Evaluar población de padres
		eval_pop(mu_r_bees, mu_r_obj, n, bee, global_id, mx, my,
			frame1, frame2, gamma, u0, v0, nx, ny, limits, cores_per_bee);
		
		// Generar población lambda
		generate_new_pop(rand1, &rand_index, rand_size, n,
			rate_alpha_r, rate_beta_r, rate_gamma_r, mu_r_obj, 
			mu_r_bees, lambda_r_bees, rand2, rand3, rand4, 
			bee, limits, num_bees);

		// Evaluar nueva población
		eval_pop(lambda_r_bees, lambda_r_obj, n, bee, global_id, mx, my,
			frame1, frame2, gamma, u0, v0, nx, ny, limits, cores_per_bee);

		// Mu + Lambda
		merge_pop(mu_r_obj, lambda_r_obj, mu_lambda_obj, mu_lambda_bees, mu_r_bees,
			lambda_r_bees, bee, n, mu_lambda_order);
	
		// Seleccionar el mejor mu
		best_mu(mu_lambda_obj, mu_lambda_order, bee, mu_r_bees, mu_r_obj, 
			mu_lambda_bees, n, num_bees);
	}
	
	// Dibujar abejas
	draw(mu_e_bees, mu_e_obj, bee, n, gamma, mx, nx);
	draw(mu_r_bees, mu_r_obj, bee, n, gamma, mx, nx);
}
