#include <torch/torch.h>
// #include <torch/serialize.h>
#include <torch/jit.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <chrono> 
#include <thread> 
#include <ctime> 
#include <inttypes.h>

#include "mnist/mnist_reader.hpp"

void terminateProcess(const char* processName) {
    char command[100];
    std::strcpy(command, "pkill -f ");
    std::strcat(command, processName);
    std::system(command);
}

// Where to find the MNIST dataset.
const char* kDataRoot = MNIST_FASHION_DATA_LOCATION;

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 50;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


struct Net : torch::nn::Module {
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 32, /*kernel_size=*/5).padding(6).stride(2)),
        conv2(torch::nn::Conv2dOptions(32, 64, /*kernel_size=*/5).padding(6).stride(2)),
        conv2_drop(torch::nn::FeatureAlphaDropoutOptions(0.3)),
        fc1(64*4*4, 120),
        fc2(120, 60),
        fc3(60, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.reshape({-1, 64*4*4});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.3, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureAlphaDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::Linear fc3;
};

template <typename DataLoader>
void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %" PRId32 " [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  // int32_t true_positives = 0;
  // int32_t false_positives = 0;
  // int32_t false_negatives = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     at::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    // Update true positives, false positives, and false negatives
    // for (int i = 0; i < pred.size(0); i++) {
    //   if (pred[i]==targets[i]) {
    //     true_positives++;
    //   } else {
    //     if (pred[i]==1) {
    //       false_positives++;
    //     } else {
    //       false_negatives++;
    //     }
    //   }
    // }
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
  // // Calculate and print Precision
  // double precision = 0.0;
  // if (true_positives + false_positives > 0) {
  //   precision = static_cast<double>(true_positives) / (true_positives + false_positives);
  // }
  // std::printf("Precision: %.3f\n", precision);

  // // Calculate and print Recall
  // double recall = 0.0;
  // if (true_positives + false_negatives > 0) {
  //   recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
  // }
  // std::printf("Recall: %.3f\n", recall);

  // // Calculate and print F1-Score
  // double f1_score = 0.0;
  // if (precision + recall > 0) {
  //   f1_score = 2.0 * (precision * recall) / (precision + recall);
  // }
  // std::printf("F1-Score: %.3f\n", f1_score);
}

// void saveModelWithArchive(const torch::nn::Module& model, const std::string& filename) {
//     torch::serialize::OutputArchive archive;
//     model.save(archive); // Save model data to the archive

//     // Open the file for writing in binary mode
//     std::ofstream outfile(filename, std::ios::binary);
//     if (!outfile.is_open()) {
//         throw std::runtime_error("Failed to open file for saving model");
//     }

//     // Serialize the archive contents into a byte stream
//     std::vector<char> buffer;
//     {
//         torch::serialize::OutputArchive buffer_archive(&buffer);
//         archive.save_to(buffer_archive);
//     }

//     // Write the buffer contents to the file
//     outfile.write(buffer.data(), buffer.size());
//     outfile.close();
// }



int main() {
    // starting the device tracking
    std::thread bash_thread([](){
        std::system("../check_device.sh");
    });

    bash_thread.detach();

    // Get the current time
    auto start_time = std::chrono::system_clock::now();
    auto start_millis = std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch()).count();
    std::time_t current_time = std::chrono::system_clock::to_time_t(start_time);

    // Format the current time as a string
    char start_time_str[100];
    std::strftime(start_time_str, sizeof(start_time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&current_time));

    // Print the current time including milliseconds
    std::cout << "Start time with milliseconds: " << start_time_str << "." << start_millis % 1000 << std::endl;
    
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Net model;
    model.to(device);

    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                            .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_dataset), kTrainBatchSize);

    auto test_dataset = torch::data::datasets::MNIST(
                            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
        torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

    torch::optim::SGD optimizer(
        model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(model, device, *test_loader, test_dataset_size);
    }

    // std::string model_path = "fashionmnist_libtorch_model.pt";
    std::cout << "Size of obj: " << sizeof(model) << " bytes" << std::endl;

    // torch::jit::save(model, model_path);
    // torch::save(model, model_path);
    // saveModelWithArchive(model, model_path);

    // torch::serialize::OutputArchive output_model_archive;
    // model.to(torch::kCPU);
    // model.save(output_model_archive);
    // output_model_archive.save_to(model_path);

    // Get the end time
    auto end_time = std::chrono::system_clock::now();
    auto end_millis = std::chrono::duration_cast<std::chrono::milliseconds>(end_time.time_since_epoch()).count();

    std::time_t full_end_time = std::chrono::system_clock::to_time_t(end_time);

    // Format the current time as a string
    char end_time_str[100];
    std::strftime(end_time_str, sizeof(end_time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&full_end_time));

    // Print the current time including milliseconds
    std::cout << "End time with milliseconds: " << end_time_str << "." << end_millis % 1000 << std::endl;

    // Calculate the difference in milliseconds
    auto duration = end_time - start_time;
    auto duration_millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << "Duration: " << duration_millis << " milliseconds" << std::endl;

    // Terminate the bash script process
    terminateProcess("../check_device.sh");
}
