#include "generator.h"

#include <unistd.h>

#include "file_io.h"
#include "allocator.h"
#include "logger.h"
#include "endianness.h"

#include "vector.h"

///////////////////////
// Linear Regression //
///////////////////////

void generate_linear_regs(int lower, int upper, float m, float y, Vector*** vecs, Vector*** targets) {
	*vecs = (Vector**)callocate(upper - lower+1, sizeof(Vector*));
	*targets = (Vector**)callocate(upper - lower+1, sizeof(Vector*));
	for (int i = lower; i <= upper; i++) {
		(*vecs)[i - lower] = vec_zero(1); (*vecs)[i - lower]->data[0] = i;
		(*targets)[i - lower] = vec_zero(1); (*targets)[i - lower]->data[0] = (m*i) + y;
	}
}

/////////
// XOR //
/////////

void generate_xor(int* lower, int* upper, Vector*** vecs, Vector*** targets) {
	*lower = 0;
	*upper = 3;
	*vecs = (Vector**)callocate(*upper - *lower+1, sizeof(Vector*));
	*targets = (Vector**)callocate(*upper - *lower+1, sizeof(Vector*));

	(*vecs)[0] = vec_zero(2);(*vecs)[0]->data[0]=-1.0f;(*vecs)[0]->data[1]=-1.0f;
	(*vecs)[1] = vec_zero(2);(*vecs)[1]->data[0]=-1.0f;(*vecs)[1]->data[1]=1.0f;
	(*vecs)[2] = vec_zero(2);(*vecs)[2]->data[0]=1.0f;(*vecs)[2]->data[1]=-1.0f;
	(*vecs)[3] = vec_zero(2);(*vecs)[3]->data[0]=1.0f;(*vecs)[3]->data[1]=1.0f;

	(*targets)[0] = vec_zero(1);(*targets)[0]->data[0] = 0.0f;
	(*targets)[1] = vec_zero(1);(*targets)[1]->data[0] = 1.0f;
	(*targets)[2] = vec_zero(1);(*targets)[2]->data[0] = 1.0f;
	(*targets)[3] = vec_zero(1);(*targets)[3]->data[0] = 0.0f;
}

///////////
// MNIST //
///////////

int _mnist_get_magic(FileData* idx_filedata) {
	FILE* fp = idx_filedata->file_pointer;
	int magic_num;
	fseek(fp, 0, SEEK_SET);
	if (fread(&magic_num, sizeof(int), 1, fp) != 1) {
		fatal("Failed to read magic number from file");
	}

	return swap_endian_int(magic_num);
}

int _mnist_get_image_count(FileData* image_filedata) {
	if (0x00000803 != _mnist_get_magic(image_filedata)) {
		fatal("Provided data file not an image file type");
	}
	FILE* fp = image_filedata->file_pointer;
	int size;
	fseek(fp, 4, SEEK_SET);
	if (fread(&size, sizeof(int), 1, fp) != 1) {
		fatal("Failed to read image count");
	}
	return swap_endian_int(size);
}

void _mnist_get_image_size(FileData* image_filedata, int size[2]) {
	if (0x00000803 != _mnist_get_magic(image_filedata)) {
		fatal("Provided data file not an image file type");
	}
	FILE* fp = image_filedata->file_pointer;
	fseek(fp, 8, SEEK_SET);
	if (fread(size, sizeof(int), 2, fp) != 2) {
		fatal("Failed to read image size");
	}
	size[0] = swap_endian_int(size[0]);
	size[1] = swap_endian_int(size[1]);
}

void _mnist_get_image(FileData* image_filedata, int img_idx, int size[2], unsigned char** data, int data_s) {
	if (0x00000803 != _mnist_get_magic(image_filedata)) {
		fatal("Provided data file not an image file type");
	}
	FILE* fp = image_filedata->file_pointer;
	int img_size = size[0]*size[1];
	if (img_size > data_s) {
		fatal("Provided data buffer is too small, need %d bytes, got %d bytes", img_size, data_s);
	}
	fseek(fp, 16+(img_size*img_idx), SEEK_SET);
	if (fread(*data, sizeof(unsigned char), img_size, fp) != (unsigned long)img_size) {
		fatal("Failed to read image data at index: %d", img_idx);
	}
}

int _mnist_get_label_count(FileData* label_filedata) {
	if (0x00000801 != _mnist_get_magic(label_filedata)) {
		fatal("Provided data file not an label file type");
	}
	FILE* fp = label_filedata->file_pointer;
	int size;
	fseek(fp, 4, SEEK_SET);
	if (fread(&size, sizeof(int), 1, fp) != 1) {
		fatal("Failed to read label count");
	}
	return swap_endian_int(size);
}

unsigned char _mnist_get_label(FileData* label_filedata, int lbl_idx) {
	if (0x00000801 != _mnist_get_magic(label_filedata)) {
		fatal("Provided data file not an label file type");
	}
	FILE* fp = label_filedata->file_pointer;
	unsigned char label;
	fseek(fp, 8+ lbl_idx, SEEK_SET);
	info("Seeked to position: %ld", ftell(fp));
	if (fread(&label, sizeof(unsigned char), 1, fp) != 1) {
		fatal("Failed to read label at index: %d", lbl_idx);
	}
	info("lbl read: %d", label);
	return label;
}

void generate_mnist(int* lower, int* upper, Vector*** vecs, Vector*** targets,
		char* train_img_path,
		char* train_lbl_path,
		char* test_img_path,
		char* test_lbl_path
		) {
	FileData* train_img = get_file_read(train_img_path);
	FileData* train_lbl = get_file_read(train_lbl_path);
	FileData* test_img = get_file_read(test_img_path);
	FileData* test_lbl = get_file_read(test_lbl_path);

	if (0x00000803 != _mnist_get_magic(train_img)) {
		fatal("Non matching magic number for training image");
	}
	if (0x00000803 != _mnist_get_magic(test_img)) {
		fatal("Non matching magic number for test image");
	}
	if (0x00000801 != _mnist_get_magic(train_lbl)) {
		fatal("Non matching magic number for training label");
	}
	if (0x00000801 != _mnist_get_magic(test_lbl)) {
		fatal("Non matching magic number for test label");
	}

	int training_img_cnt = _mnist_get_image_count(train_img);
	int test_img_cnt = _mnist_get_image_count(test_img);
	int training_img_size[2]; _mnist_get_image_size(train_img, training_img_size);
	int test_img_size[2]; _mnist_get_image_size(test_img, test_img_size);
	int training_lbl_cnt = _mnist_get_label_count(train_lbl);
	int test_lbl_cnt = _mnist_get_label_count(test_lbl);

	info("Training image count: %d, size of %dx%d", training_img_cnt, training_img_size[0], training_img_size[1]);
	info("Test image count: %d, size of %dx%d", test_img_cnt, test_img_size[0], test_img_size[1]);
	info("Training label count: %d", training_lbl_cnt);
	info("Test label count: %d", test_lbl_cnt);

	int img_size = training_img_size[0]*training_img_size[1];
	int img_idx = 2;
	unsigned char* img_buf = (unsigned char*)allocate(img_size);
	_mnist_get_image(train_img, img_idx, training_img_size, &img_buf, img_size);
	unsigned char img_label = _mnist_get_label(train_lbl, img_idx);
	info("Example training image at idx: %d of a %d", img_idx, img_label);
	for (int i = 0; i < training_img_size[0]; i++) {
		for (int j = 0; j < training_img_size[1]; j++) {
			if (img_buf[(i*training_img_size[0]) + j]) {
				printr("X");
			} else {
				printr(" ");
			}
		}
		newline();
	}
	newline();

	close_file(train_img);
	close_file(train_lbl);
	close_file(test_img);
	close_file(test_lbl);
}
