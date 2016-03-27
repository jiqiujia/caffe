#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb", "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv){
	::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the image mean_value of a set of images given by"
	        " a leveldb/lmdb\n"
			"Usage:\n"
			"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 2 || argc > 3) {
	    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
	    return 1;
	}

	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[1], db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	//load first datum
	Datum datum;
	datum.ParseFromString(cursor->value());
	if(DecodeDatumNative(&datum)){
		LOG(INFO) << "Decoding Datum";
	}

/*	BlobProto sum_blob;
	sum_blob.setnum(1);
	sum_blob.set_channels(datum.channels());
	sum_blob.set_height(1);
	sum_blob.set_width(1);
	for (int i=0; i<datum.channels(); i++){
		sum_blob.add_data(0.);
	}
*/
	double* count = new double[datum.channels()];
	double* sum = new double[datum.channels()];
	

	LOG(INFO) << "Starting Iteration";
	int tmp = 0;
	while(cursor->valid()){
		Datum datum;
		datum.ParseFromString(cursor->value());
		DecodeDatumNative(&datum);

		const std::string& data = datum.data();
		if(data.size()!=0){
			for(int i = 0; i<datum.channels(); i++){		
				int k = datum.height()*datum.width();
				for(int j = 0; j<k; j++){
					sum[i] += data[i*k+j];
				}
				count[i] += k;
			}
		}else if(datum.float_data_size() !=0){
			for(int i = 0; i<datum.channels(); i++){		
				int k = datum.height()*datum.width();
				for(int j = 0; j<k; j++){
					sum[i] += datum.float_data(i*k+j);
				}
				count[i] += k;
			}
		}
		tmp++;
		if (tmp%1000==0)
			LOG(INFO) << "processed " << tmp << "files.";
	}
	for(int i = 0;i<datum.channels();i++){
		sum[i] /= count[i];
		LOG(INFO)<<"mean_value channel [" << i << "]" << sum[i];
	}
#else
	LOG(FATAL) <<"This tool requires OpenCV; compile with USE_OPENCV.";
#endif
	return 0;
}
