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
	if (fread(&label, sizeof(unsigned char), 1, fp) != 1) {
		fatal("Failed to read label at index: %d", lbl_idx);
	}
	return label;
}

void generate_mnist(int* lower, int* upper, Vector*** vecs, Vector*** targets,
		char* img_path,
		char* lbl_path
		) {
	FileData* img_fdata = get_file_read(img_path);
	FileData* lbl_fdata = get_file_read(lbl_path);

	if (0x00000803 != _mnist_get_magic(img_fdata)) {
		fatal("Non matching magic number for image data");
	}
	if (0x00000801 != _mnist_get_magic(lbl_fdata)) {
		fatal("Non matching magic number for label data");
	}

	int img_cnt = _mnist_get_image_count(img_fdata);
	int img_size[2]; _mnist_get_image_size(img_fdata, img_size);
	int lbl_cnt = _mnist_get_label_count(lbl_fdata);

	if (img_cnt != lbl_cnt) {
		fatal("Mismatching image count to label count");
	}

	int img_buf_size = img_size[0]*img_size[1];
	unsigned char* img_buf = (unsigned char*)allocate(img_buf_size);
	*lower = 0;
	*upper = lbl_cnt;
	*vecs = (Vector**)allocate(sizeof(Vector*)*lbl_cnt);
	*targets = (Vector**)allocate(sizeof(Vector*)*lbl_cnt);

	for (int t = 0; t < lbl_cnt; t++) {
		unsigned char img_lbl = _mnist_get_label(lbl_fdata, t);
		_mnist_get_image(img_fdata, t, img_size, &img_buf, img_buf_size);

		(*vecs)[t] = vec_zero(img_buf_size);
		(*targets)[t] = vec_zero(10);

		for (int i = 0; i < img_buf_size; i++) {
			(*vecs)[t]->data[i] = img_buf[i] / 255.0f;
		}
		(*targets)[t]->data[img_lbl] = 1.0f;
	}

	close_file(img_fdata);
	close_file(lbl_fdata);
	deallocate(img_buf);
}
