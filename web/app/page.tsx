// app/page.tsx
import { spawn } from "child_process";
import { ArrowUpIcon, ArrowDownIcon } from "lucide-react";
import path from "path";

interface PredictionResult {
  current_price: number;
  predicted_price: number;
  price_change: number;
  last_update: string;
}

async function predictStock(): Promise<{
  result?: PredictionResult;
  error?: string;
}> {
  return new Promise((resolve) => {
    const scriptPath = path.join(process.cwd(), "scripts", "predict_stock.py");
    console.log("스크립트 경로:", scriptPath);

    const pythonProcess = spawn("python3", [scriptPath]);

    let dataString = "";
    let errorString = "";

    pythonProcess.stdout.on("data", (data) => {
      console.log("Python stdout:", data.toString());
      dataString += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("Python stderr:", data.toString());
      errorString += data.toString();
    });

    pythonProcess.on("error", (error) => {
      console.error("Process Error:", error);
      resolve({ error: `프로세스 실행 오류: ${error.message}` });
    });

    pythonProcess.on("close", (code) => {
      console.log("Python process exited with code:", code);
      console.log("Final stdout:", dataString);
      console.log("Final stderr:", errorString);

      if (code !== 0) {
        resolve({ error: `예측 실패 (종료 코드: ${code})\n${errorString}` });
        return;
      }

      try {
        const result = JSON.parse(dataString);
        if (result.error) {
          resolve({ error: result.error });
        } else {
          resolve({ result: result });
        }
      } catch (err) {
        resolve({
          error: `결과 파싱 실패: ${err.message}\n데이터: ${dataString}`,
        });
      }
    });
  });
}

export default async function Home() {
  console.log("페이지 렌더링 시작");
  const { result, error } = await predictStock();
  console.log("예측 결과:", result);
  console.log("에러:", error);

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("en-US").format(price);
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-sans">
      <div className="w-full max-w-2xl">
        <h1 className="text-2xl font-bold mb-8">나스닥 지수 예측</h1>

        {error ? (
          <div className="p-4 bg-red-100 border border-red-400 rounded-md text-red-700 whitespace-pre-wrap">
            오류 발생: {error}
          </div>
        ) : result ? (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-6 bg-white rounded-lg shadow-md">
                <h2 className="text-lg font-semibold text-gray-600 mb-2">
                  현재 지수
                </h2>
                <p className="text-3xl font-bold">
                  {formatPrice(result.current_price)}
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  마지막 업데이트: {result.last_update}
                </p>
              </div>

              <div className="p-6 bg-white rounded-lg shadow-md">
                <h2 className="text-lg font-semibold text-gray-600 mb-2">
                  예상 지수
                </h2>
                <p className="text-3xl font-bold">
                  {formatPrice(result.predicted_price)}
                </p>
                <div
                  className={`flex items-center mt-2 ${
                    result.price_change >= 0 ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {result.price_change >= 0 ? (
                    <ArrowUpIcon className="w-4 h-4 mr-1" />
                  ) : (
                    <ArrowDownIcon className="w-4 h-4 mr-1" />
                  )}
                  <span className="font-semibold">
                    {Math.abs(result.price_change)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-blue-50 border border-blue-200 rounded-md text-sm text-blue-800">
              * 이 예측은 과거 데이터를 기반으로 한 참고 자료이며, 실제 시장
              움직임과 다를 수 있습니다.
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
