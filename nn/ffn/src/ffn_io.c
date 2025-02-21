#include "ffn_io.h"

#include <stdio.h>

#include "allocator.h"

#include "optimizer.h"

void ffn_export_model(FFNModel* model, FileData* datafile) {
	FILE* fptr = datafile->file_pointer;
	size_t param_bytes = model->papool->base.data_size;
	size_t layer_cnt = model->init_data->hidden_size;

	fwrite(&layer_cnt, sizeof(size_t), 1, fptr);
	for (int l = 0; l < (int)layer_cnt; l++) {
		LayerData* l_data = model->init_data->hidden_layers[l];
		fwrite(&(l_data->node_cnt), sizeof(size_t), 1, fptr);
		fwrite(&(l_data->l_type), sizeof(LayerType), 1, fptr);
		fwrite(&(l_data->fn_type), sizeof(ActivationFNEnum), 1, fptr);
		fwrite(&(l_data->w_init_type), sizeof(IniterEnum), 1, fptr);
		fwrite(&(l_data->b_init_type), sizeof(IniterEnum), 1, fptr);
	}
	fwrite(&(model->init_data->cost_fn_enum), sizeof(CostFnEnum), 1, fptr);
	fwrite(&(model->optimizer->type), sizeof(OptimizerTypeEnum), 1, fptr);
	fwrite(model->optimizer->config, resolve_optimizer_config_size(model->optimizer->type), 1, fptr);
	fwrite(&(model->batch_type), sizeof(BatchTypeEnum), 1, fptr);
	fwrite(&(model->batch_size), sizeof(size_t), 1, fptr);

	fwrite(&param_bytes, sizeof(size_t), 1, fptr);
	fwrite(model->papool->base.dptr, sizeof(char), param_bytes, fptr);
}

FFNModel* ffn_import_model(FileData* datafile) {
	FILE* fptr = datafile->file_pointer;

	FFNModel* model = ffn_new_model();
	size_t layer_cnt, optimizer_config_size, batch_size, param_bytes;
	LayerData* l_data = (LayerData*)allocate(sizeof(LayerData));
	CostFnEnum cost_fn_type;
	OptimizerTypeEnum optimizer_type;
	void* optimizer_config;
	BatchTypeEnum batch_type;


	fread(&layer_cnt, sizeof(size_t), 1, fptr);

	for (int l = 0; l < (int)layer_cnt; l++) {
		fread(&(l_data->node_cnt), sizeof(size_t), 1, fptr);
		fread(&(l_data->l_type), sizeof(LayerType), 1, fptr);
		fread(&(l_data->fn_type), sizeof(ActivationFNEnum), 1, fptr);
		fread(&(l_data->w_init_type), sizeof(IniterEnum), 1, fptr);
		fread(&(l_data->b_init_type), sizeof(IniterEnum), 1, fptr);
		
		ffn_add_layer(model, l_data);
	}

	fread(&cost_fn_type, sizeof(CostFnEnum), 1, fptr);
	fread(&optimizer_type, sizeof(OptimizerTypeEnum), 1, fptr);
	optimizer_config_size = resolve_optimizer_config_size(optimizer_type);
	optimizer_config = allocate(optimizer_config_size);
	fread(optimizer_config, optimizer_config_size, 1, fptr);
	fread(&batch_type, sizeof(BatchTypeEnum), 1, fptr);
	fread(&batch_size, sizeof(size_t), 1, fptr);
	fread(&param_bytes, sizeof(size_t), 1, fptr);

	ffn_set_cost_fn(model, cost_fn_type);
	ffn_set_batch_type(model, batch_type);
	ffn_set_batch_size(model, batch_size);
	ffn_set_optimizer(model, nn_build_optimizer(optimizer_type, optimizer_config));
	ffn_finalize(model);

	fread(model->papool->base.dptr, sizeof(char), param_bytes, fptr);

	deallocate(l_data);
	deallocate(optimizer_config);

	return model;
}
