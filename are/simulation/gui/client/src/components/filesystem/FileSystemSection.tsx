// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

import ArticleIcon from "@mui/icons-material/Article";
import CloseIcon from "@mui/icons-material/Close";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DownloadIcon from "@mui/icons-material/Download";
import FolderIcon from "@mui/icons-material/Folder";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import ImageIcon from "@mui/icons-material/Image";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import TableChartIcon from "@mui/icons-material/TableChart";
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  Typography,
} from "@mui/material";
import {
  TreeItemCheckbox,
  TreeItemContent,
  TreeItemDragAndDropOverlay,
  TreeItemGroupTransition,
  TreeItemIcon,
  TreeItemIconContainer,
  TreeItemLabel,
  TreeItemProvider,
  TreeItemRoot,
  useTreeItem,
  UseTreeItemParameters,
} from "@mui/x-tree-view";
import { RichTreeView } from "@mui/x-tree-view/RichTreeView";
import { forwardRef, Suspense, useContext, useMemo, useState } from "react";
import {
  graphql,
  PreloadedQuery,
  usePreloadedQuery,
  useQueryLoader,
} from "react-relay";
import SessionIdContext from "../../contexts/SessionIdContext";
import Image from "../common/Image";
import FileUploadDialog from "./FileUploadDialog";
import { FileSystemSectionGetDownloadQuery } from "./__generated__/FileSystemSectionGetDownloadQuery.graphql";
import { FileSystemSectionGetFileContentQuery } from "./__generated__/FileSystemSectionGetFileContentQuery.graphql";
import { FileNode } from "./types";

const FileContentQuery = graphql`
  query FileSystemSectionGetFileContentQuery(
    $sessionId: String!
    $filesystemAppName: String!
    $filePath: String!
  ) {
    getFileContent(
      sessionId: $sessionId
      filesystemAppName: $filesystemAppName
      filePath: $filePath
    )
  }
`;

const DownloadContentQuery = graphql`
  query FileSystemSectionGetDownloadQuery(
    $sessionId: String!
    $filesystemAppName: String!
    $filePath: String!
  ) {
    downloadFile(
      sessionId: $sessionId
      filesystemAppName: $filesystemAppName
      filePath: $filePath
    )
  }
`;

const DIALOG_MAX_WIDTH = 800;

interface FileSystemSectionProps {
  files: FileNode;
  filesystemAppName: string;
}

interface TreeItemProps
  extends
    Omit<UseTreeItemParameters, "rootRef">,
    Omit<React.HTMLAttributes<HTMLLIElement>, "onFocus"> {}

interface NodeData {
  name: string;
  type: string;
  path: string;
  fileExtension?: string;
}

interface TreeItem {
  id: string;
  label: string;
  children: TreeItem[];
  nodeData: NodeData;
}

/**
 * Component representing a section of the file system, displaying files and directories.
 *
 * @param files - The root file node containing the file structure.
 * @param filesystemAppName - The name of the filesystem application.
 */
function FileSystemSection({
  files,
  filesystemAppName,
}: FileSystemSectionProps) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [dialogData, setDialogData] = useState<NodeData | null>(null);
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);

  const [fileContentQueryReference, loadFileContentQuery] =
    useQueryLoader<FileSystemSectionGetFileContentQuery>(FileContentQuery);
  const [downloadContentQueryReference, loadDownloadContentQuery] =
    useQueryLoader<FileSystemSectionGetDownloadQuery>(DownloadContentQuery);

  const sessionId = useContext(SessionIdContext);

  /**
   * Handles the click event on a tree node, loading file content and download data if the node is a file.
   *
   * @param item - The tree item that was clicked.
   */
  const handleNodeClick = (item: TreeItem) => {
    if (!item || !item.nodeData) return;

    if (item.nodeData.type === "file") {
      setDialogData(item.nodeData);
      loadFileContentQuery({
        sessionId,
        filesystemAppName,
        filePath: item.id,
      });
      loadDownloadContentQuery({
        sessionId,
        filesystemAppName,
        filePath: item.id,
      });
      setIsDialogOpen(true);
    }
  };

  /**
   * Closes the dialog and resets the dialog data.
   */
  const handleClose = () => {
    setIsDialogOpen(false);
    setDialogData(null);
  };

  /**
   * Custom TreeItem component for rendering tree nodes with additional functionality.
   */
  const TreeItem = forwardRef(function TreeItem(
    props: TreeItemProps,
    ref: React.Ref<HTMLLIElement>,
  ) {
    const { id, itemId, label, disabled, children, ...other } = props;

    const {
      getRootProps,
      getContentProps,
      getIconContainerProps,
      getCheckboxProps,
      getLabelProps,
      getGroupTransitionProps,
      getDragAndDropOverlayProps,
      status,
      publicAPI,
    } = useTreeItem({ id, itemId, children, label, disabled, rootRef: ref });

    const item: TreeItem = publicAPI.getItem(itemId);

    return (
      <TreeItemProvider id={id} itemId={itemId}>
        <TreeItemRoot {...getRootProps(other)}>
          <TreeItemContent
            {...getContentProps()}
            onClick={(e: React.MouseEvent) => {
              getContentProps().onClick(e);
              handleNodeClick(item);
            }}
          >
            <TreeItemIconContainer {...getIconContainerProps()}>
              <TreeItemIcon status={status} />
            </TreeItemIconContainer>
            <Stack
              direction={"row"}
              flexGrow={1}
              spacing={1}
              alignItems={"center"}
            >
              {item.nodeData.type === "file" ? (
                <FileIcon fileExtension={item.nodeData.fileExtension ?? ""} />
              ) : status.expanded ? (
                <FolderOpenIcon />
              ) : (
                <FolderIcon />
              )}
              <TreeItemCheckbox {...getCheckboxProps()} />
              <TreeItemLabel {...getLabelProps()} />
            </Stack>
            <TreeItemDragAndDropOverlay {...getDragAndDropOverlayProps()} />
          </TreeItemContent>
          {children && (
            <TreeItemGroupTransition {...getGroupTransitionProps()} />
          )}
        </TreeItemRoot>
      </TreeItemProvider>
    );
  });

  /**
   * Memoized computation of tree items from the file structure.
   */
  const items = useMemo(() => {
    if (!files || !files.children) return [];

    // Create a root node
    const rootNode = {
      id: "/",
      label: "/",
      children: files.children.map((child: FileNode) => createTreeItems(child)),
      nodeData: {
        name: "/",
        type: "directory",
        path: "/",
      },
    };

    return [rootNode];
  }, [files]);

  return (
    <>
      <Dialog open={isDialogOpen} onClose={handleClose} maxWidth="md">
        <DialogTitle>
          <Stack direction="row" alignItems="center">
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              <Stack direction="row" spacing={1} alignItems="center">
                <FileIcon fileExtension={dialogData?.fileExtension ?? ""} />
                <Typography variant="h6">{dialogData?.name}</Typography>
              </Stack>
            </Typography>
            <IconButton onClick={handleClose}>
              <CloseIcon />
            </IconButton>
          </Stack>
        </DialogTitle>
        <DialogContent>
          <Stack spacing={1}>
            <Typography fontWeight={"bold"}>Details</Typography>
            <Card>
              <CardContent>
                <Typography>
                  <b>Name: </b>
                  {dialogData?.name || ""}
                </Typography>
                <Typography>
                  <b>Type: </b>
                  {dialogData?.type || ""}
                </Typography>
                <Typography>
                  <b>Path: </b>
                  {dialogData?.path || ""}
                </Typography>
              </CardContent>
            </Card>
            <Typography fontWeight={"bold"}>Content</Typography>
            <Card>
              <CardContent>
                <Box
                  display={"flex"}
                  maxWidth={DIALOG_MAX_WIDTH}
                  width={"100%"}
                  overflow={"auto"}
                  flexDirection={"column"}
                  justifyContent={"center"}
                  alignItems={"center"}
                >
                  <Suspense fallback={<CircularProgress />}>
                    {fileContentQueryReference != null ? (
                      <FileContent
                        fileContentQueryReference={fileContentQueryReference}
                        fileExtension={dialogData?.fileExtension ?? ""}
                      />
                    ) : null}
                  </Suspense>
                </Box>
              </CardContent>
            </Card>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Suspense fallback={<Button disabled>Preparing Download</Button>}>
            {downloadContentQueryReference != null ? (
              <DownloadContent
                downloadContentQueryReference={downloadContentQueryReference}
                fileName={dialogData?.name ?? "file"}
                fileExtension={dialogData?.fileExtension ?? ""}
              />
            ) : null}
          </Suspense>
        </DialogActions>
      </Dialog>

      <FileUploadDialog
        open={isUploadDialogOpen}
        onClose={() => setIsUploadDialogOpen(false)}
        filesystemAppName={filesystemAppName}
        onUploadSuccess={() => {
          setIsUploadDialogOpen(false);
        }}
      />

      <Stack spacing={2}>
        <Box>
          <Button
            variant="contained"
            startIcon={<CloudUploadIcon />}
            onClick={() => setIsUploadDialogOpen(true)}
            sx={{ mb: 2 }}
          >
            Upload File
          </Button>
        </Box>
        <RichTreeView
          items={items}
          slots={{ item: TreeItem }}
          defaultExpandedItems={items?.length > 0 ? ["/"] : undefined}
        />
      </Stack>
    </>
  );
}

const FileContent = ({
  fileContentQueryReference,
  fileExtension,
}: {
  fileContentQueryReference: PreloadedQuery<FileSystemSectionGetFileContentQuery>;
  fileExtension: string;
}) => {
  const data = usePreloadedQuery(FileContentQuery, fileContentQueryReference);
  if (!data.getFileContent) {
    return <div>No file content available</div>;
  }
  if (["png", "jpg", "jpeg"].includes(fileExtension)) {
    return (
      <Image
        src={`data:image/${fileExtension};base64,${data.getFileContent}`}
        width={"100%"}
      />
    );
  }

  return <Typography>{data.getFileContent}</Typography>;
};

const DownloadContent = ({
  downloadContentQueryReference,
  fileName,
  fileExtension,
}: {
  downloadContentQueryReference: PreloadedQuery<FileSystemSectionGetDownloadQuery>;
  fileName: string;
  fileExtension: string;
}) => {
  const data = usePreloadedQuery(
    DownloadContentQuery,
    downloadContentQueryReference,
  );

  const downloadUrl = useMemo(() => {
    if (!data.downloadFile) {
      return null;
    }

    try {
      const byteCharacters = atob(data.downloadFile);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const mimeType = getMimeType(fileExtension);
      const blob = new Blob([byteArray], { type: mimeType });

      return URL.createObjectURL(blob);
    } catch (error) {
      console.error("Error processing download file:", error);
      return null;
    }
  }, [data, fileExtension]);

  if (!downloadUrl) {
    return <Typography>No download URL available</Typography>;
  }

  return (
    <Button
      download={fileName}
      href={downloadUrl}
      startIcon={<DownloadIcon />}
      variant={"contained"}
    >
      Download File
    </Button>
  );
};

const getFileExtension = (filePath: string) => {
  return filePath.split(".").pop()?.toLowerCase();
};

const createTreeItems = (node: FileNode, path = ""): TreeItem => {
  const nodePath = `${path}/${node.name}`;
  return {
    id: nodePath,
    label: node.name,
    children: node.children
      ? node.children.map((child) => createTreeItems(child, nodePath))
      : [],
    nodeData: {
      name: node.name,
      type: node.type,
      path: nodePath,
      fileExtension: getFileExtension(nodePath),
    },
  };
};

const getMimeType = (fileExtension: string) => {
  switch (fileExtension) {
    case "txt":
      return "text/plain";
    case "pdf":
      return "application/pdf";
    case "png":
      return "image/png";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "gif":
      return "image/gif";
    case "json":
      return "application/json";
    default:
      return "application/octet-stream";
  }
};

const FileIcon = ({ fileExtension }: { fileExtension: string }) => {
  switch (fileExtension) {
    case "pdf":
      return <PictureAsPdfIcon />;
    case "png":
    case "jpg":
    case "jpeg":
    case "gif":
      return <ImageIcon />;
    case "csv":
      return <TableChartIcon />;
    default:
      return <ArticleIcon />;
  }
};

export default FileSystemSection;
