#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <iostream>

#include "cprof/dependencies.hpp"

DependencyTracking::DependencyTracking(){}

DependencyTracking &DependencyTracking::instance() {
    static DependencyTracking a;
    return a;
  }

void DependencyTracking::memory_ptr_create(uintptr_t addr_ptr){
    mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{this->connection_url}};
    bsoncxx::builder::stream::document document{};
    auto collection = conn[this->database]["pointers"];

    document << "ptr" << (long)addr_ptr;
    document << "actions" << bsoncxx::builder::stream::open_array << bsoncxx::builder::stream::close_array;

    collection.insert_one(document.view());
}

void DependencyTracking::action_on_ptr(const CUpti_CallbackData *cbInfo, uintptr_t addr_ptr){
    using namespace bsoncxx::builder::stream;
    mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{this->connection_url}};
    bsoncxx::builder::stream::document document{};
    
    auto pointer_collection = conn[this->database]["pointers"];
    auto action_collection = conn[this->database]["actions"];
    
    bsoncxx::builder::stream::document action_doc{};
    action_doc << "correlationId" << (long)cbInfo->correlationId;
    action_doc << "functionName" << std::string(cbInfo->functionName);
    action_doc << "symbolName" << std::string(cbInfo->symbolName);
    bsoncxx::stdx::optional<mongocxx::result::insert_one> result =
    action_collection.insert_one(action_doc.view());

    pointer_collection.update_one(document << "ptr" << (long)addr_ptr << finalize,
    document << "$push" << open_document <<
      "actions" << result->inserted_id() << close_document << finalize);
}

void DependencyTracking::annotate_times(uint16_t correlationId, std::chrono::nanoseconds start_point, std::chrono::nanoseconds end_point){
    using namespace bsoncxx::builder::stream;    
    mongocxx::instance inst{};
    mongocxx::client conn{mongocxx::uri{"mongodb://dominic:1080isgreat@ds127439.mlab.com:27439/dependency-graph"}};
    bsoncxx::builder::stream::document document{};
    auto action_collection = conn["dependency-graph"]["actions"];

    action_collection.update_many(document << "correlationId" << (long)correlationId << finalize,
    document << "$set" << open_document <<
      "start_time" << start_point.count() <<
      "end_time" << end_point.count() << 
      close_document << finalize);
}