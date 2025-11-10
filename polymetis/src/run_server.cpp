// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "polymetis/polymetis_server.hpp"
#include "real_time.hpp"
#include "torch_server_ops.hpp"

struct RunServerArgs {
  std::string &server_address;
  bool log_to_csv;
};

void *RunServer(void *run_server_args_ptr) {
  RunServerArgs &run_server_args =
      *(static_cast<RunServerArgs *>(run_server_args_ptr));
  std::string &server_address = run_server_args.server_address;

  // Instantiate service
  PolymetisControllerServerImpl service(run_server_args.log_to_csv);

  // Build service
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());

  // Start server
  spdlog::info("Server listening on {}", server_address);
  server->Wait();

  return NULL;
}

int main(int argc, char **argv) {
  // Parse inputs
  InputParser input(argc, argv);

  if (input.cmdOptionExists("-h")) {
    spdlog::info("Usage: polymetis_server [OPTION]");
    spdlog::info("Starts a controller manager server.");
    spdlog::info("  -h        Help");
    spdlog::info("  -r        Use real-time");
    spdlog::info("  -s <ip>   Change server address");
    spdlog::info("  -p <port> Change server port");
    spdlog::info("  -l        Log Controller Output to CSV");
    return 0;
  }

  bool use_real_time = false;
  if (input.cmdOptionExists("-r")) {
    use_real_time = true;
  }

  std::string ip = "0.0.0.0";
  if (input.cmdOptionExists("-s")) {
    ip = input.getCmdOption("-s");
  }
  std::string port = "50051";
  if (input.cmdOptionExists("-p")) {
    port = input.getCmdOption("-p");
  }
  std::string server_address = ip + ":" + port;

  bool log_to_csv = false;
  if (input.cmdOptionExists("-l")) {
    log_to_csv = true;
  }
  spdlog::info("Using real time: {}", use_real_time);
  spdlog::info("Using server address: {}", server_address);
  spdlog::info("Using controller logging: {}", log_to_csv);

  // Start real-time thread
  RunServerArgs run_server_args{server_address, log_to_csv};
  void *run_server_args_ptr = static_cast<void *>(&run_server_args);

  if (!use_real_time) {
    RunServer(run_server_args_ptr);
  } else {
    create_real_time_thread(RunServer, run_server_args_ptr);
  }

  return 0;
}