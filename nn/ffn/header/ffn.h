#ifndef FFN_H
/** \file */
#define FFN_H

#include "booltype.h"

#include "vector.h"

#include "activation.h"
#include "cost.h"
#include "initer.h"
#include "optimizer.h"

#include "ffn_mempool.h"
#include "ffn_init.h"

/**
 * \struct FFNModel
 * \brief A Feed forward network model
 * 
 * A Feed forward network model composing of dense/passthrough
 * layers to process data
 */
typedef struct {
	FFNParams* init_data; /**< Initialization data */
	FFNParameterPool* papool; /**< Paramters */
	FFNPropagationPool* prpool; /**< Propagation values */
	FFNGradientPool* gpool; /**< Gradient values */
	FFNIntermediatePool* ipool; /**< Intermediate values */
	BatchTypeEnum batch_type; /**< Batching algorithm type */
	size_t batch_size; /**< Batch size */
	Optimizer* optimizer; /**< Optimizer struct \see Optimizer */
	booltype immutable; /**< Network's immutability */
} FFNModel;

// Creation

/**
 * \brief Allocates a new FFN model
 *
 * \return A Feedforward network model uninitialized
 */
FFNModel* ffn_new_model();

// Memory Management

/**
 * \brief Deallocate a FFN model
 *
 * \param model The model to deallocate
 */
void ffn_deallocate_model(FFNModel*);

// Initialization

/**
 * Add a dense layer to a model
 *
 * \param model Model to add the layer to
 * \param size Size of the layer
 * \param activation_fn Activation function to be used for the layer
 * \param w_init Weight initializer for the layer
 * \param b_init Bias initializer for the layer
 *
 * \see ActivationFNEnum
 * \see IniterEnum
 */
void ffn_add_dense(FFNModel*, size_t, ActivationFNEnum, IniterEnum, IniterEnum);

/**
 * Add a pass through layer to the model
 *
 * \param model Model to add the layer to
 * \param activation_fn Activation functio to pass the preactivation through
 *
 * \see ActivationFNEnum
 */
void ffn_add_passthrough(FFNModel*, ActivationFNEnum);

/**
 * Add a layer from a LayerData*
 *
 * \param model Model to add the layer to
 * \param layer_data Layer data to add
 *
 * \see LayerData
 */
void ffn_add_layer(FFNModel*, LayerData*);

/**
 * Set the cost function of a model
 *
 * \param model Model to set the cost function
 * \param cost_fn_type The type of cost function to set to
 *
 * \see CostFnEnum
 */
void ffn_set_cost_fn(FFNModel*, CostFnEnum);

/**
 * Set the optimizer of a model
 *
 * \param model Model to set the optimizer
 * \param optimizer The optimizer struct to be used
 *
 * \see Optimizer
 */
void ffn_set_optimizer(FFNModel*, Optimizer*);

/**
 * Set the batch type of a model
 *
 * \param model Model to set the batch type
 * \param batch_type The type of batching algorithm to use
 *
 * \see BatchTypeEnum
 */
void ffn_set_batch_type(FFNModel*, BatchTypeEnum);

/**
 * Set the batch size of a model
 * \note Can be omitted if using FullBatch or Stochastic batch type
 *
 * \param model Model to set the batch size
 * \param batch_size The Batch size to set to
 */
void ffn_set_batch_size(FFNModel*, size_t);

/**
 * Finalize a FFN model
 * \note Finalized FFN model cant be re initalized nor have configuration changed
 *
 * \param model Model to finalize
 */
void ffn_finalize(FFNModel*);

// Running and Training

/**
 * Run a data point through a FFN model
 *
 * \param model Model to be used to run
 * \param input Input data point
 *
 * \return An output vector
 */
Vector* ffn_run(FFNModel*, Vector*);

/**
 * Trains a dataset to a FFN model
 *
 * \param model Model to train
 * \param input Input data
 * \param target Target output \note target[i] correspond to input[i]
 * \param dataset_size Size of the whole dataset
 * \param learning_rate Learning rate of the model
 * \param max_t Maximum amount of data points that can be used
 *
 * \return A float containing the average loss of the model
 */
float ffn_train(FFNModel*, Vector**, Vector**, size_t, float, int);

#endif
